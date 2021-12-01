from types import SimpleNamespace

import torch


def get_mask_from_lengths(lengths, max_len=None):
    if not max_len:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        self.dictionary = dictionary
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)

    def keys(self):
        return self.dictionary.keys()

    def __getitem__(self, key):
        return self.dictionary[key]


def get_vocoder_generator(state_dict: dict):
    generator = {}
    for key, value in state_dict.items():
        if "generator" in key:
            generator[key[10:]] = value
    return generator