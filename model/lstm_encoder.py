import torch
import torch.nn as nn
from model.encoder_config import EncoderConfig
from model.encoder_output import EncoderOutput


class LSTMEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig, pad_id: int):
        super().__init__()
        self._cfg = cfg
        self._pad_id = pad_id
        self._embed = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=pad_id)
        self._lstm = nn.LSTM(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )

    def forward(self, src_ids: torch.Tensor, src_mask: torch.Tensor) -> EncoderOutput:
        x = self._embed(src_ids)
        states, (h, c) = self._lstm(x)
        return EncoderOutput(states=states, final_hidden=h, final_cell=c, mask=src_mask)
