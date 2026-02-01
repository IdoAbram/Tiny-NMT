import torch
import torch.nn as nn


class LossComputer:
    def __init__(self, pad_id: int):
        self._pad_id = pad_id
        self._ce = nn.CrossEntropyLoss(ignore_index=pad_id)

    def compute(self, logits: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        targets = self._targets(tgt_ids)
        flat_logits, flat_targets = self._flatten(logits, targets)
        return self._ce(flat_logits, flat_targets)

    def _targets(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        # targets = המילה הבאה: מהעמודה 1 ועד הסוף
        return tgt_ids[:, 1:]

    def _flatten(self, logits: torch.Tensor, targets: torch.Tensor):
        b, t, v = logits.shape
        return logits.reshape(b * t, v), targets.reshape(b * t)
