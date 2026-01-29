import torch
from data.sequence_padder import SequencePadder
from data.mask_builder import MaskBuilder
from data.batch import Batch


class BatchCollator:
    def __init__(self, pad_id: int):
        self._padder = SequencePadder(pad_id)
        self._masker = MaskBuilder()

    def collate(self, items: list[tuple[torch.Tensor, torch.Tensor]]) -> Batch:
        src_list, tgt_list = self._split(items)
        src_ids, src_mask = self._pad_and_mask(src_list)
        tgt_ids, tgt_mask = self._pad_and_mask(tgt_list)
        return Batch(src_ids, src_mask, tgt_ids, tgt_mask)

    def _split(self, items):
        src = [x[0] for x in items]
        tgt = [x[1] for x in items]
        return src, tgt

    def _pad_and_mask(self, seqs: list[torch.Tensor]):
        lengths = self._padder.lengths(seqs)
        padded = self._padder.pad(seqs)
        mask = self._masker.build(lengths, padded.shape[1])
        return padded, mask
