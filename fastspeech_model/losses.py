import torch

from fastspeech_model.utils import get_mask_from_lengths


class DurationLoss():
    def __call__(self, *, log_duration_pred, duration_target, mask):
        log_duration_target = torch.log(duration_target + 1)
        loss = torch.nn.functional.mse_loss(log_duration_pred, log_duration_target, reduction='none')
        return (loss * mask.squeeze(dim=-1)).sum() / mask.sum()


class L2MelLoss():
    def __call__(self, *, spec_pred, spec_target, spec_target_len, pad_value=0):
        spec_target.requires_grad = False
        max_len = spec_target.shape[2]

        if max_len < spec_pred.shape[2]:
            # Predicted len is larger than reference
            # Need to slice
            spec_pred = spec_pred.narrow(2, 0, max_len)
        elif max_len > spec_pred.shape[2]:
            # Need to do padding
            pad_amount = max_len - spec_pred.shape[2]
            spec_pred = torch.nn.functional.pad(spec_pred, (0, pad_amount), value=pad_value)
            max_len = spec_pred.shape[2]

        mask = ~get_mask_from_lengths(spec_target_len, max_len=max_len)
        mask = mask.expand(spec_target.shape[1], mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)

        mel_loss = torch.nn.functional.mse_loss(spec_pred, spec_target, reduction='none')
        return (mel_loss * (~mask)).sum() / (~mask).sum()


def masked_loss(loss_f, pred, gt, mask):
    # loss_f - torch loss function with reduction='none
    # pred & gt has [B, T] shape, mask - [B, T, 1]
    loss = loss_f(pred, gt) * mask.squeeze(dim=-1)
    return torch.sum(loss) / torch.sum(mask)
