# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import pytorch_lightning as pl

from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from torch import optim

from fastspeech_model.losses import L2MelLoss, DurationLoss, masked_loss
from fastspeech_model.modules import FastSpeech2Encoder, FastSpeech2Decoder, VarianceAdaptor


class FastSpeech2Model(pl.LightningModule):
    """FastSpeech 2 model used to convert from text (phonemes) to mel-spectrograms."""

    def __init__(self, cfg):
        super().__init__()

        self.pitch = cfg.add_pitch_predictor
        self.energy = cfg.add_energy_predictor
        self.duration_coeff = cfg.duration_coeff
        self.second_stage_start = cfg.second_stage_start

        self.encoder = FastSpeech2Encoder(**cfg.encoder)
        self.mel_decoder = FastSpeech2Decoder(**cfg.decoder)
        self.variance_adapter = VarianceAdaptor(**cfg.variance_adaptor)


        self.loss = L2MelLoss()
        self.mseloss = torch.nn.MSELoss(reduction='none')
        self.durationloss = DurationLoss()

        self.vocoder = instantiate(self.cfg.vocoder)
        checkpoint = torch.load(self.cfg.vocoder_pretrain_path)
        generator_state_dict = get_vocoder_generator(checkpoint['state_dict'])
        # generator_state_dict = checkpoint['generator']
        self.vocoder.load_state_dict(generator_state_dict)
        self.vocoder.remove_weight_norm()
        self.vocoder.eval()

        self.vocoder_sample_rate = self.cfg.sample_rate


    def configure_optimizers(self):
        self.optim = optim.Adam(params=itertools.chain(self.encoder.parameters(), self.mel_decoder.parameters(), self.variance_adapter.parameters()),
                                lr=0.0001)
        # self.optim_g = instantiate(self._cfg.optim,
        #                           params=itertools.chain(self.encoder.parameters(), self.mel_decoder.parameters(), self.variance_adapter.parameters()))

        """
        self.scheduler_g = NoamAnnealing(
            optimizer=self.optim_g,
            warmup_steps=self._cfg.sched.warmup_steps,
            min_lr=self._cfg.sched.min_lr,
            d_model=self._cfg.sched.d_model,
            max_steps=50000
        )
        
        sch1_dict = {
            'scheduler': self.scheduler_g,
            'interval': 'step',
        }
        """
        #return [self.optim_g], [sch1_dict]
        return self.optim

    def forward(self, *, text, text_length, spec_len=None, durations=None, pitch=None, energies=None):
        encoded_text, encoded_text_mask = self.encoder(text=text, text_length=text_length)
        aligned_text, log_dur_preds, pitch_preds, energy_preds, spec_len = self.variance_adapter(
            x=encoded_text,
            x_len=text_length,
            dur_target=durations,
            pitch_target=pitch,
            energy_target=energies,
            spec_len=spec_len,
        )
        mel = self.mel_decoder(decoder_input=aligned_text, lengths=spec_len)
        return mel, log_dur_preds, pitch_preds, energy_preds, encoded_text_mask

    def training_step(self, batch, batch_idx):
        spec, spec_len, t, tl, durations, pitch, energies = batch

        mel, log_dur_preds, pitch_preds, energy_preds, encoded_text_mask = self(
            text=t, text_length=tl, spec_len=spec_len, durations=durations, pitch=pitch, energies=energies
        )

        mel_loss = self.loss(
            spec_pred=mel.transpose(1, 2), spec_target=spec, spec_target_len=spec_len, pad_value=-11.52
        )
        dur_loss = self.durationloss(
            log_duration_pred=log_dur_preds, duration_target=durations.float(), mask=encoded_text_mask
        )
        dur_loss *= self.duration_coeff
        pitch_loss = masked_loss(self.mseloss, pitch_preds, pitch, encoded_text_mask)
        energy_loss = masked_loss(self.mseloss, energy_preds, energies, encoded_text_mask)

        reconstruction_loss = mel_loss + dur_loss + pitch_loss + energy_loss

        if self.global_step >= self.second_stage_start:
            real_out, real_features, generated_out, generated_features = self.discriminator(spec, mel.transpose(1, 2))
            feature_matching_loss = self.feature_loss(real_features, generated_features)
            adversarial_generator_loss = torch.mean((1 - generated_out) ** 2)
            lambda_fm = reconstruction_loss.item() / feature_matching_loss.item()
            total_loss = reconstruction_loss + adversarial_generator_loss + lambda_fm * feature_matching_loss
        else:
            total_loss = reconstruction_loss

        self.manual_backward(total_loss)
        self.optim_g.step()
        self.optim_g.zero_grad()

        if self.global_step >= self.second_stage_start:
            real_out, _, generated_out, _ = self.discriminator(spec, mel.detach().transpose(1, 2))

            d_loss = torch.mean((1 - real_out) ** 2) + torch.mean(generated_out ** 2)

            self.manual_backward(d_loss)
            self.optim_d.step()
            self.optim_d.zero_grad()

        schedulers = self.lr_schedulers()
        if schedulers is not None:
            sch1, sch2 = schedulers
            sch1.step()
            if self.global_step >= self.second_stage_start:
                sch2.step()

        if batch_idx == 0:
            lens = ["fl", spec_len, (torch.exp(log_dur_preds) - 1).sum(dim=1)]
            self.log_wavs(spec, mel, lens, "train")

        self.log(name="train_mel_loss", value=mel_loss)
        self.log(name="train_dur_loss", value=dur_loss)
        self.log(name="train_pitch_loss", value=pitch_loss)
        self.log(name="train_energy_loss", value=energy_loss)

        if self.global_step >= self.second_stage_start:
            self.log(name="train_fm_loss", value=feature_matching_loss)
            self.log(name="train_g_loss", value=adversarial_generator_loss)
            self.log(name="train_d_loss", value=d_loss)
        self.log(name="train_loss", value=reconstruction_loss, sync_dist=True)
        self.log("r_loss", reconstruction_loss, prog_bar=True, logger=False, sync_dist=True)

    def training_epoch_end(self, outputs):
        if self.log_train_images and self.logger is not None and self.logger.experiment is not None \
                and self.log_train == "images":
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            spec_target, spec_predict = outputs[0]["outputs"]
            tb_logger.add_image(
                "train_mel_target",
                plot_spectrogram_to_numpy(spec_target[0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = spec_predict[0].data.cpu().numpy()
            tb_logger.add_image(
                "train_mel_predicted", plot_spectrogram_to_numpy(spec_predict.T), self.global_step, dataformats="HWC",
            )
            self.log_train_images = False

            return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        spec, spec_len, t, tl, durations, pitch, energies = batch

        mel, log_dur_preds, pitch_preds, energy_preds, mask = self(text=t, text_length=tl, spec_len=spec_len)
        loss = self.loss(spec_pred=mel.transpose(1, 2), spec_target=spec, spec_target_len=spec_len, pad_value=-11.52)

        pitch_loss = masked_loss(self.mseloss, pitch_preds, pitch, mask)
        energy_loss = masked_loss(self.mseloss, energy_preds, energies, mask)
        dur_loss = self.durationloss(
            log_duration_pred=log_dur_preds, duration_target=durations.float(), mask=mask
        )

        for _ in range(3):
            self.log(name="val_loss", value=loss)
            self.log(name="val_pitch_loss", value=pitch_loss)
            self.log(name="val_energy_loss", value=energy_loss)
            self.log(name="val_duration_loss", value=dur_loss)

        if batch_idx == 0:
            lens = ["fl", spec_len, (torch.round(torch.exp(log_dur_preds)) - 1).sum(dim=1)]
            self.log_wavs(spec, mel, lens, "validation")

        return {"val_loss": loss}

    def log_wavs(self, gt_mel, pred_mel, lens, split):
        # takes only one from gt wav and gt mel cuz it always work great
        lens = (lens[1][0] + 0.5).int(), lens[2].int() #, (lens[0][0] + 0.5).int()
        lens = list(map(lambda x: torch.maximum(x, torch.tensor([7]).cuda()), lens))
        with torch.no_grad():
            generated_on_gt_mel = self.vocoder(x=gt_mel[:1, :, :lens[0]])  # first sample from batch
            n = 8 if split == "validation" else 4
            for i in range(n):
                generated_on_predicted_mel = self.vocoder(x=pred_mel[i: i + 1, :lens[1][i]].transpose(1, 2))
                self.logger.experiment.add_audio(f"{split} generated in predicted mel {i}",
                                                 generated_on_predicted_mel.detach().cpu().numpy(),
                                                 global_step=self.global_step, sample_rate=self.vocoder_sample_rate)
        #self.logger.experiment.add_audio(f"{split} gt wav", gt_wav[:1, :lens[2]].cpu().numpy(),
        #                                 global_step=self.global_step, sample_rate=self.vocoder_sample_rate)
        self.logger.experiment.add_audio(f" {split} generated on gt mel", generated_on_gt_mel.detach().cpu().numpy(),
                                         global_step=self.global_step, sample_rate=self.vocoder_sample_rate)

    def validation_epoch_end(self, outputs):
        if self.logger is not None and self.logger.experiment is not None and self.log_train == "images":
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break
            _, spec_target, spec_predict = outputs[0].values()
            tb_logger.add_image(
                "val_mel_target",
                plot_spectrogram_to_numpy(spec_target[0].data.cpu().numpy()),
                self.global_step,
                dataformats="HWC",
            )
            spec_predict = spec_predict[0].data.cpu().numpy()
            tb_logger.add_image(
                "val_mel_predicted", plot_spectrogram_to_numpy(spec_predict.T), self.global_step, dataformats="HWC",
            )
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()  # This reduces across batches, not workers!
        self.log('val_loss', avg_loss, sync_dist=True)

        self.log_train_images = True

