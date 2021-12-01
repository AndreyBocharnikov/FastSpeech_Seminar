import os
import json
from typing import Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


class FastSpeech2DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.filename_train = cfg.filenames[0]
        self.bsz_train = cfg.batch_sizes[0]
        self.filename_val = cfg.filenames[1]
        self.bsz_val = cfg.batch_sizes[1]
        self.preprocessed_path = cfg.preprocessed_path
        self.phone_mapping = cfg.phone_mapping

    def setup(self, stage: Optional[str] = None):
        self.train = FastSpeech2Dataset(self.filename_train, self.preprocessed_path, self.phone_mapping)
        self.val = FastSpeech2Dataset(self.filename_val, self.preprocessed_path, self.phone_mapping)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.bsz_train)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.bsz_val)


class FastSpeech2Dataset(Dataset):
    def __init__(self, filename, preprocessed_path, phone_mapping):
        self.preprocessed_path = preprocessed_path

        with open(phone_mapping) as f:
            self.phone_mapping = json.load(f)
            self.phone_mapping = {v: int(k) for k, v in self.phone_mapping.items()}

        self.pad_value = len(self.phone_mapping)
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(filename)

    def process_meta(self, filename):
        full_path = os.path.join(self.preprocessed_path, filename)
        all_phonems = set()
        with open(full_path, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                assert len(t) >= 2, t
                all_phonems.update(t[1:-1].split())
                raw_text.append(r)
            return name, speaker, text, raw_text

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        phone = np.array([self.phone_mapping[phone] for phone in self.text[idx].split()])

        """ MING ORIGINAL
        mel_path = os.path.join(
             self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        """
        mel_path = os.path.join(self.preprocessed_path, "mel", "{}_mel.npy".format(basename))
        pitch_path = os.path.join(self.preprocessed_path, "pitch", "{}_pitch.npy".format(basename))
        energy_path = os.path.join(self.preprocessed_path, "energy_phones", "{}_energy.npy".format(basename))
        duration_path = os.path.join(self.preprocessed_path, "durations", "{}_dur.npy".format(basename))

        mel = np.load(mel_path)
        pitch = np.load(pitch_path)
        energy = np.load(energy_path)
        duration = np.load(duration_path)

        return phone, mel, pitch, energy, duration

    def collate_fn(self, batch):
        phones, mels, pitches, energies, durations = zip(*batch)
        # phones, wavs, pitches, energies, durations = zip(*batch)

        max_mel_len = max([mel.shape[0] for mel in mels])
        # max_wav_len = max([wav.shape[1] for wav in wavs])
        max_duration = max([duration.shape[0] for duration in durations])
        mel_lens = torch.tensor(np.array([mel.shape[0] for mel in mels]))
        # wav_lens = torch.tensor(np.array([wav.shape[1] for wav in wavs]))
        phones_lens = torch.tensor(np.array([duration.shape[0] for duration in phones]))

        padded_phones, padded_pitches, padded_energies, padded_durations = [], [], [], []
        for phone, pitch, energy, duration in zip(phones, pitches, energies, durations):
            cur_duration = pitch.shape[0]
            pad = (0, max_duration - cur_duration)
            assert phone.shape == pitch.shape == energy.shape == duration.shape
            padded_phones.append(torch.nn.functional.pad(torch.tensor(phone), pad, value=self.pad_value))
            padded_pitches.append(torch.nn.functional.pad(torch.tensor(pitch), pad))
            padded_energies.append(torch.nn.functional.pad(torch.tensor(energy), pad))
            padded_durations.append(torch.nn.functional.pad(torch.tensor(duration), pad))

        padded_mels = []
        for mel in mels:
            cur_mel_len = mel.shape[0]
            pad = (0, max_mel_len - cur_mel_len, 0, 0)
            padded_mels.append(torch.nn.functional.pad(torch.tensor(mel).T, pad))

        """
        padded_wavs = []
        for wav in wavs:
            cur_wav_len = wav.shape[1]
            pad = (0, max_wav_len - cur_wav_len)
            padded_wavs.append(torch.nn.functional.pad(wav[0], pad))
        """

        mels = torch.stack(padded_mels).float()
        # wavs = torch.stack(padded_wavs).float()
        phones = torch.stack(padded_phones).long()
        pitches = torch.stack(padded_pitches).float()
        energies = torch.stack(padded_energies).float()
        durations = torch.stack(padded_durations).float()

        return mels, mel_lens, phones, phones_lens, durations, pitches, energies
        # return wavs, wav_lens, phones, phones_lens, durations, pitches, energies

    def __len__(self):
        return len(self.text)
