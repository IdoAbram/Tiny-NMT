import torch
from data.constants import Constants


class MaskBuilder:
    def build(self, lengths: list[int], max_len: int) -> torch.Tensor:
        mask = torch.zeros((len(lengths), max_len), dtype=torch.bool)
        for i, n in enumerate(lengths):
            mask[i, :n] = True
        return mask

    def max_len(self, lengths: list[int]) -> int:
        return max(lengths) if lengths else 0
