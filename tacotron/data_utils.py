import json
import numpy as np
import torch
import torch.utils.data
import os
import torchaudio
import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        #assert dataset_type in ["tacotron", 'fastspeech']
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.language = hparams.language
        if self.language == "ru":
            with open(hparams.phone_mapping_path) as f:
                self.phone_mapping = {v: k for k, v in json.load(f).items()}

        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        audiopath = os.path.join('/content/RUSLAN', audiopath) + '.wav'
        if self.language == "en":
            text = self.get_text(text)
        else:
            text = self.get_text_ruslan(text)
        mel, energy = self.get_mel(audiopath)

        return text, mel, audiopath, energy

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = torchaudio.load(filename)
            if sampling_rate != self.stft.sampling_rate:
                resampler = torchaudio.transforms.Resample(sampling_rate, self.stft.sampling_rate)
                audio = resampler(audio)[0]
                # raise ValueError("{} SR doesn't match target {} SR".format(
                #    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec, energy = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec, energy

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def get_text_ruslan(self, text):
        return torch.tensor([int(self.phone_mapping[phone]) for phone in text.split()], dtype=torch.long)

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class ComputeFeaturesCollate:
    def __init__(self, hparams):
        self.n_mel_channels = hparams.n_mel_channels

    def __call__(self, batch):
        batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)

        max_text_length = max([text.size(0) for text, *_ in batch])
        max_mel_length = max([mel.size(1) for _, mel, *_ in batch])

        texts_padded = torch.LongTensor(len(batch), max_text_length).zero_()
        text_lengths = torch.LongTensor(len(batch)).zero_()
        mels_padded = torch.FloatTensor(len(batch), self.n_mel_channels, max_mel_length).zero_()
        mel_lengths = torch.LongTensor(len(batch)).zero_()
        energy_padded = torch.FloatTensor(len(batch), 1, max_mel_length).zero_()

        file_ids = []

        for i, (text, mel, file_id, energy) in enumerate(batch):
            texts_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)
            mels_padded[i, :, :mel.size(1)] = mel
            mel_lengths[i] = mel.size(1)
            energy_padded[i, :, :mel.size(1)] = energy
            file_ids.append(file_id)

        max_len = torch.max(text_lengths.data).item()
        return (texts_padded, text_lengths, mels_padded, max_len, mel_lengths), file_ids, energy_padded


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths
