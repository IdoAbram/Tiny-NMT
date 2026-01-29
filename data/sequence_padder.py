import torch
from data.constants import Constants


class SequencePadder:
    def __init__(self, pad_id: int):
        self._pad_id = pad_id

    def pad(self, sequences: list[torch.Tensor]) -> torch.Tensor:
        max_len = self._max_len(sequences)
        return torch.stack([self._pad_one(s, max_len) for s in sequences])

    def lengths(self, sequences: list[torch.Tensor]) -> list[int]:
        return [len(s) for s in sequences]

    def _max_len(self, sequences: list[torch.Tensor]) -> int:
        return max((len(s) for s in sequences), default=0)

    def _pad_one(self, seq: torch.Tensor, max_len: int) -> torch.Tensor:
        out = torch.full((max_len,), self._pad_id, dtype=torch.long)
        out[: len(seq)] = seq.to(torch.long)
        return out
