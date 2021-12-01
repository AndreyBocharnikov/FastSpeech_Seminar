import numpy as np
import torch
import torch.nn as nn

from fastspeech_model.submodules import VariancePredictor, LengthRegulator, FFTransformer
from fastspeech_model.utils import get_mask_from_lengths


class FastSpeech2Encoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_layers=4,
        n_attn_heads=2,
        d_attn_head=256,
        d_inner=1024,
        kernel_size=9,
        dropout=0.1,
        attn_dropout=0.1,
        n_embed=84,
        padding_idx=83,
    ):
        """
        FastSpeech 2 encoder. Converts phoneme sequence to the phoneme hidden sequence.
        Consists of a phoneme embedding lookup, positional encoding, and feed-forward
        Transformer blocks (4 by default).

        Args:
            d_model: Model input (embedding) dimension. Defaults to 256.
            n_layers: Number of feed-forward Transformer layers in the encoder. Defaults to 4.
            n_attn_heads: Number of attention heads for the feed-forward Transformer in the encoder.
                Defaults to 2.
            d_attn_head: Dimensionality of the attention heads. Defaults to 256.
            d_inner: Encoder hidden dimension. Defaults to 1024.
            kernel_size: Encoder Conv1d kernel size (kernel_size, 1). Defaults to 9.
            dropout: Encoder feed-forward Transformer dropout. Defaults to 0.1.
            attn_dropout: Encoder attention dropout. Defaults to 0.1.
            n_embed: Embedding input dim, should match number of tokens. Defaults to 84.
            padding_idx: Padding token index. Deafaults to 83.
        """
        super().__init__()

        self.encoder = FFTransformer(
            n_layer=n_layers,
            n_head=n_attn_heads,
            d_model=d_model,
            d_head=d_attn_head,
            d_inner=d_inner,
            kernel_size=(kernel_size, 1),
            dropout=dropout,
            dropatt=attn_dropout,
            embed_input=True,
            n_embed=n_embed,
            padding_idx=padding_idx,
        )

    def forward(self, text, text_length):
        return self.encoder(text, seq_lens=text_length)


class VarianceAdaptor(nn.Module):
    def __init__(
        self,
        d_model=256,
        dropout=0.2,
        dur_d_hidden=256,
        dur_kernel_size=3,
        pitch=True,
        log_pitch=True,
        n_f0_bins=256,
        pitch_kernel_size=3,
        pitch_min=80.0,
        pitch_max=800.0,
        pitch_boundaries_path=None,
        use_predefined_boundaries=False,
        energy=True,
        n_energy_bins=256,
        energy_kernel_size=3,
        energy_min=0.0,
        energy_max=600.0,
        supplementary_first=True
    ):
        """
        FastSpeech 2 variance adaptor, which adds information like duration, pitch, etc. to the phoneme encoding.
        Sets of conv1D blocks with ReLU and dropout.

        Args:
            d_model: Input and hidden dimension. Defaults to 256 (default encoder output dim).
            dropout: Variance adaptor dropout. Defaults to 0.2.
            dur_d_hidden: Hidden dim of the duration predictor. Defaults to 256.
            dur_kernel_size: Kernel size for the duration predictor. Defaults to 3.
            pitch (bool): Whether or not to use the pitch predictor.
            log_pitch (bool): If True, uses log pitch. Defaults to True.
            n_f0_bins: Number of F0 bins for the pitch predictor. Defaults to 256.
            pitch_kernel_size: Kernel size for the pitch predictor. Defaults to 3.
            pitch_min: Defaults to 80.0.
            pitch_max: Defaults to 800.0.
            pitch_d_hidden: Hidden dim of the pitch predictor.
            energy (bool): Whether or not to use the energy predictor.
            n_energy_bins: Number of energy bins. Defaults to 256.
            energy_kernel_size: Kernel size for the energy predictor. Defaults to 3.
            energy_min: Defaults to 0.0.
            energy_max: Defaults to 600.0.
        """
        super().__init__()

        # -- Duration Setup --
        self.duration_predictor = VariancePredictor(
            d_model=d_model, d_inner=dur_d_hidden, kernel_size=dur_kernel_size, dropout=dropout
        )
        self.length_regulator = LengthRegulator()

        pitch_bins = torch.linspace(start=pitch_min, end=pitch_max, steps=n_f0_bins - 1)

        self.register_buffer("pitch_bins", pitch_bins)
        self.pitch_predictor = VariancePredictor(
            d_model=d_model, d_inner=n_f0_bins, kernel_size=pitch_kernel_size, dropout=dropout
        )
        self.pitch_lookup = nn.Embedding(n_f0_bins, d_model)

        self.register_buffer(  # Linear scale bins
            "energy_bins", torch.linspace(start=energy_min, end=energy_max, steps=n_energy_bins - 1)
        )
        self.energy_predictor = VariancePredictor(
            d_model=d_model, d_inner=n_energy_bins, kernel_size=energy_kernel_size, dropout=dropout,
        )
        self.energy_lookup = nn.Embedding(n_energy_bins, d_model)

    def phonems_to_mels(self, x, log_dur_preds, dur_target, spec_len):
        if dur_target is not None:
            dur_out = self.length_regulator(x, dur_target)
        else:
            dur_preds = torch.clamp_min(torch.round(torch.exp(log_dur_preds)) - 1, 0).long()
            if not torch.sum(dur_preds, dim=1).bool().all():
                dur_preds += 1
            dur_out = self.length_regulator(x, dur_preds)
            spec_len = torch.sum(dur_preds, dim=1)
        return dur_out, spec_len

    def forward(self, *, x, x_len, dur_target=None, pitch_target=None, energy_target=None, spec_len=None):
        """
        Args:
            x: Input from the encoder.
            x_len: Length of the input.
            dur_target:  Duration targets for the duration predictor. Needs to be passed in during training.
            pitch_target: Pitch targets for the pitch predictor. Needs to be passed in during training.
            energy_target: Energy targets for the energy predictor. Needs to be passed in during training.
            spec_len: Target spectrogram length. Needs to be passed in during training.
        """
        # Duration predictions (or ground truth) fed into Length Regulator to
        # expand the hidden states of the encoder embedding
        log_dur_preds = self.duration_predictor(x)
        log_dur_preds.masked_fill_(~get_mask_from_lengths(x_len), 0)
        # Output is Batch, Time

        dur_out = x
        out = dur_out.clone()

        mask = get_mask_from_lengths(x_len if self.supplementary_first else spec_len)
        pitch_preds = self.pitch_predictor(dur_out)
        if pitch_target is not None:
            pitch_out = self.pitch_lookup(torch.bucketize(pitch_target, self.pitch_bins))
        else:
            pitch_out = self.pitch_lookup(torch.bucketize(pitch_preds.detach(), self.pitch_bins))
        out += pitch_out
        out *= mask.unsqueeze(dim=2)

        # Energy
        energy_preds = self.energy_predictor(dur_out)
        if energy_target is not None:
            energy_out = self.energy_lookup(torch.bucketize(energy_target, self.energy_bins))
        else:
            energy_out = self.energy_lookup(torch.bucketize(energy_preds.detach(), self.energy_bins))
        out += energy_out
        out *= mask.unsqueeze(dim=2)

        out, spec_len = self.phonems_to_mels(out, log_dur_preds, dur_target, spec_len)
        out *= get_mask_from_lengths(spec_len).unsqueeze(-1)

        return out, log_dur_preds, pitch_preds, energy_preds, spec_len


class FastSpeech2Decoder(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_out=80,
        n_layers=4,
        n_attn_heads=2,
        d_attn_head=256,
        d_inner=1024,
        kernel_size=9,
        dropout=0.1,
        attn_dropout=0.1,
    ):
        """
        FastSpeech 2 mel-spectrogram decoder. Converts adapted hidden sequence to a mel-spectrogram sequence.
        Consists of four feed-forward Transformer blocks by default.

        Args:
            d_model: Input dimension. Defaults to 256.
            d_out: Dimensionality of output. Defaults to 80.
            n_layers: Number of feed-forward Transformer layers in the mel-spec decoder. Defaults to 4.
            n_attn_heads: Number of attention heads for the feed-forward Transformer. Defaults to 2.
            d_attn_head: Dimensionality of the attention heads. Defaults to 256.
            d_inner: Mel-spec decoder hidden dimension. Defaults to 1024.
            kernel_size: Mel-spec decoder Conv1d kernel size (kernel_size, 1). Defaults to 9.
            dropout: Mel-spec decoder feed-forward Transformer dropout. Defaults to 0.1.
            attn_dropout: Mel-spec decoder attention dropout. Defaults to 0.1.
        """
        super().__init__()

        self.decoder = FFTransformer(
            n_layer=n_layers,
            n_head=n_attn_heads,
            d_model=d_model,
            d_head=d_attn_head,
            d_inner=d_inner,
            kernel_size=(kernel_size, 1),
            dropout=dropout,
            dropatt=attn_dropout,
            embed_input=False,
        )
        self.linear = nn.Linear(d_model, d_out)

    def forward(self, decoder_input, lengths):
        decoder_out, _ = self.decoder(decoder_input, lengths)
        mel_out = self.linear(decoder_out)
        return mel_out
