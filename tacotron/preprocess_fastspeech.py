import argparse
import os
import random
import json
from copy import deepcopy

import tgt
import librosa
import numpy as np
import pyworld as pw
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio

from data_utils import TextMelLoader, ComputeFeaturesCollate
from hparams import create_hparams
from train import load_model, warm_start_model


class Preprocessor:
    def __init__(self, need_tacotron, hparams):
        self.in_dir = hparams.raw_path
        self.out_dir = hparams.preprocessed_path

        self.sampling_rate = hparams.sampling_rate
        self.hop_length = hparams.hop_length

        self.pitch_phoneme_averaging = True
        self.energy_phoneme_averaging = True

        self.pitch_normalization = True
        self.energy_normalization = True

        if need_tacotron:
            self.dataset = TextMelLoader(hparams.meta_phones, hparams)
            collate_fn = ComputeFeaturesCollate(hparams)

            self.loader = DataLoader(self.dataset, num_workers=1, shuffle=False,
                                     sampler=None,
                                     batch_size=hparams.batch_size, pin_memory=False,
                                     drop_last=False, collate_fn=collate_fn)

            model = load_model(hparams)
            self.model = warm_start_model(hparams.checkpoint_path, model, [])

        self.sample_phones_mapping = self.load_sample_phones_mapping(hparams.meta_phones)
        print(f"{len(self.sample_phones_mapping)} saples")

    def load_sample_phones_mapping(self, path):
        sample_phones_mapping = dict()
        with open(path, encoding='utf-8') as f:
            for i, line in enumerate(f):
                sample_path, phones = line.strip().split("|")
                name = sample_path.split('/')[-1].split('.')[0]
                sample_phones_mapping[name] = phones
        return sample_phones_mapping

    def dp_alignment(self, attention):
        """
        Here you are given with an attention prob matrix
            with a shape of N_frames X N_graphemes

        You should compute the optimal way according to the formula above with DP.
        With the optimal way (sequence j_i) computed you should return durations.
        So, duration of j-th gratheme == #(i: j_i == j) -- number of frames at which
            our optimal way is 'stuck' at the grapheme.
        These durations, as an integer numpy array should be returned from the function.
        """

        # !!!!!!!!!!!!!!!!!!!!!!
        n, m = attention.shape
        dp = np.full_like(attention, np.NINF)
        came_from = np.zeros(attention.shape)

        def get_log(value):
            return np.log(value)

        for i in range(dp.shape[0]):
            for j in range(dp.shape[1]):
                if i == 0 and j == 0:
                    dp[i][i] = get_log(attention[i][j])
                elif j == 0:
                    dp[i][j] = dp[i - 1][j] + get_log(attention[i][j])
                elif i == 0:
                    continue
                else:
                    if dp[i - 1][j - 1] > dp[i - 1][j]:
                        dp[i][j] = dp[i - 1][j - 1] + get_log(attention[i][j])
                        came_from[i][j] = 1
                    else:
                        dp[i][j] = dp[i - 1][j] + get_log(attention[i][j])

        cur_pos = [n - 1, m - 1]
        cur_graphem = m - 1
        cur_duration = 1
        durations = []
        while cur_pos[0] != 0 or cur_pos[1] != 0:
            if came_from[cur_pos[0]][cur_pos[1]]:
                durations.append(cur_duration)
                cur_graphem -= 1
                cur_pos[0] -= 1
                cur_pos[1] -= 1
                cur_duration = 1
            else:
                cur_duration += 1
                cur_pos[0] -= 1

        durations.append(cur_duration)
        durations = durations[::-1]
        assert sum(durations) == n
        assert len(durations) == m
        return durations

    def get_paths(self, basename):
        dur_path = os.path.join(self.out_dir, 'durations', basename + "_dur.npy")
        mel_path = os.path.join(self.out_dir, 'mel', basename + '_mel.npy')
        energy_path = os.path.join(self.out_dir, 'energy_frame', basename + '_energy.npy')
        return dur_path, energy_path, mel_path

    def compute_durations(self):
        os.makedirs((os.path.join(self.out_dir, "durations")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_frame")), exist_ok=True)

        with torch.no_grad():
            for batch, file_ids, energy_padded in tqdm(self.loader):
                batch = [elem.cuda()
                         if torch.is_tensor(elem)
                         else elem
                         for elem in batch]
                texts, text_lengths, mels, _, mel_lengths = batch
                *_, alignments = self.model(batch)
                alignments = alignments.cpu().numpy()

                for j, alignment in enumerate(alignments):
                    basename = file_ids[j].split('/')[-1].split('.')[0]
                    dur_path, energy_path, mel_path = self.get_paths(basename)

                    durations = self.dp_alignment(alignment[: mel_lengths[j], : text_lengths[j]])
                    mel = mels[j, :, :mel_lengths[j]].T.cpu().numpy()
                    energy = energy_padded[j, :, :mel_lengths[j]].cpu().numpy()
                    assert mel.shape[0] == sum(durations)
                    assert energy.shape[1] == sum(durations)
                    np.save(dur_path, durations)
                    np.save(mel_path, mel)
                    np.save(energy_path, energy)

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_phones")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, speaker in enumerate(os.listdir(self.in_dir)):
            speakers[speaker] = i
            for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker))):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                if basename not in self.sample_phones_mapping:
                    print('not in', basename)
                    continue
                ret = self.process_utterance(speaker, basename)
                info, pitch, energy, n = ret
                out.append(info)  # todo, change info

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))

                n_frames += n
        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy_phones"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )
        """
        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size:]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out
        """

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        # basename += "_RUSLAN"
        phones = self.sample_phones_mapping[basename].split()

        dur_path, energy_path, mel_path = self.get_paths(basename)
        duration = np.load(dur_path)

        assert len(duration) == len(phones), dur_path

        text = "{" + " ".join(phones) + "}"
        raw_text = "unused"

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav.astype(np.float32)

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        pitch = pitch[: sum(duration)]
        assert np.sum(pitch != 0) > 1, dur_path

        mel_spectrogram = np.load(mel_path)
        energy = np.load(energy_path).squeeze()
        mel_spectrogram = mel_spectrogram[: sum(duration), :]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos: pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos: pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        pitch_filename = "{}_pitch.npy".format(basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}_energy.npy".format(basename)
        np.save(os.path.join(self.out_dir, "energy_phones", energy_filename), energy)

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[0],
        )

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value


if __name__ == "__main__":
    # right_order()
    # check_distribution()

    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    args = parser.parse_args()

    hparams = create_hparams(args.hparams)
    need_tacotron = False

    preprocessor = Preprocessor(need_tacotron, hparams)
    if need_tacotron:
        preprocessor.compute_durations()
    else:
        preprocessor.build_from_path()