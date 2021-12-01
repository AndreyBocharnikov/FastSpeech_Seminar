import itertools

import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from torch import optim

from fastspeech_model.losses import L2MelLoss, DurationLoss, masked_loss
from fastspeech_model.modules import FastSpeech2Encoder, FastSpeech2Decoder, VarianceAdaptor
from fastspeech_model.utils import get_vocoder_generator
from hifigan_generator import Generator


class FastSpeech2Model(pl.LightningModule):
    """FastSpeech 2 model used to convert from text (phonemes) to mel-spectrograms."""

    def __init__(self, cfg):
        super().__init__()

        self.duration_coeff = cfg.duration_coeff

        self.encoder = FastSpeech2Encoder(**cfg.encoder)
        self.mel_decoder = FastSpeech2Decoder(**cfg.decoder)
        self.variance_adapter = VarianceAdaptor(**cfg.variance_adaptor)

        self.loss = L2MelLoss()
        self.mseloss = torch.nn.MSELoss(reduction='none')
        self.durationloss = DurationLoss()

        self.vocoder = Generator(**cfg.vocoder)
        checkpoint = torch.load(cfg.vocoder_pretrain_path, map_location='cpu')
        generator_state_dict = get_vocoder_generator(checkpoint['state_dict'])

        self.vocoder.load_state_dict(generator_state_dict)
        self.vocoder.remove_weight_norm()
        self.vocoder.eval()

        self.vocoder_sample_rate = cfg.sample_rate

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

        if batch_idx == 0:
            lens = ["fl", spec_len, (torch.exp(log_dur_preds) - 1).sum(dim=1)]
            self.log_wavs(spec, mel, lens, "train")

        self.log(name="train_mel_loss", value=mel_loss)
        self.log(name="train_dur_loss", value=dur_loss)
        self.log(name="train_pitch_loss", value=pitch_loss)
        self.log(name="train_energy_loss", value=energy_loss)

        self.log(name="train_loss", value=reconstruction_loss, prog_bar=True, sync_dist=True) #logger=False,
        return reconstruction_loss

    def validation_step(self, batch, batch_idx):
        spec, spec_len, t, tl, durations, pitch, energies = batch

        mel, log_dur_preds, pitch_preds, energy_preds, mask = self(text=t, text_length=tl, spec_len=spec_len)
        loss = self.loss(spec_pred=mel.transpose(1, 2), spec_target=spec, spec_target_len=spec_len, pad_value=-11.52)

        pitch_loss = masked_loss(self.mseloss, pitch_preds, pitch, mask)
        energy_loss = masked_loss(self.mseloss, energy_preds, energies, mask)
        dur_loss = self.durationloss(
            log_duration_pred=log_dur_preds, duration_target=durations.float(), mask=mask
        )

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
            n = 4 if split == "validation" else 4
            for i in range(n):
                generated_on_predicted_mel = self.vocoder(x=pred_mel[i: i + 1, :lens[1][i]].transpose(1, 2))
                self.logger.experiment.add_audio(f"{split} generated in predicted mel {i}",
                                                 generated_on_predicted_mel.detach().cpu().numpy(),
                                                 global_step=self.global_step, sample_rate=self.vocoder_sample_rate)
        #self.logger.experiment.add_audio(f"{split} gt wav", gt_wav[:1, :lens[2]].cpu().numpy(),
        #                                 global_step=self.global_step, sample_rate=self.vocoder_sample_rate)
        self.logger.experiment.add_audio(f" {split} generated on gt mel", generated_on_gt_mel.detach().cpu().numpy(),
                                         global_step=self.global_step, sample_rate=self.vocoder_sample_rate)

