import torch


class Batch:
    def __init__(self, src_ids: torch.Tensor, src_mask: torch.Tensor,
                 tgt_ids: torch.Tensor, tgt_mask: torch.Tensor | None = None):
        self.src_ids = src_ids
        self.src_mask = src_mask
        self.tgt_ids = tgt_ids
        self.tgt_mask = tgt_mask
