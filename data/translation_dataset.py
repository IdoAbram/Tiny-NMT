import torch
from torch.utils.data import Dataset

from data.sequence_encoder import SequenceEncoder


class TranslationDataset(Dataset):
    def __init__(self, rows: list[dict], encoder: SequenceEncoder):
        self._rows = rows
        self._encoder = encoder

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int):
        row = self._rows[idx]
        src_text = row["src"]
        tgt_text = row["tgt"]

        src_ids = self._encoder.encode_source(src_text)
        tgt_ids = self._encoder.encode_target(tgt_text)

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )
