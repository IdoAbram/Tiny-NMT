import torch
import torch.nn as nn

from model.decoder_config import DecoderConfig
from model.decoder_state import DecoderState
from model.decoder_output import DecoderOutput
from model.bahdanau_attention import BahdanauAttention


class LSTMDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig, pad_id: int, attention: BahdanauAttention, enc_dim: int):
        super().__init__()
        self._embed = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=pad_id)
        self._attn = attention
        self._cell = nn.LSTMCell(cfg.emb_dim + enc_dim, cfg.hidden_dim)
        self._out = nn.Linear(cfg.hidden_dim + enc_dim, cfg.vocab_size)
        self._drop = nn.Dropout(cfg.dropout)

    def init_state(self, batch_size: int, device: torch.device) -> DecoderState:
        h = torch.zeros(batch_size, self._cell.hidden_size, device=device)
        c = torch.zeros(batch_size, self._cell.hidden_size, device=device)
        return DecoderState(h, c)

    def step(self, prev_ids: torch.Tensor, state: DecoderState,
             enc_states: torch.Tensor, src_mask: torch.Tensor) -> DecoderOutput:
        emb = self._drop(self._embed(prev_ids))
        ctx, attn = self._attn(state.h, enc_states, src_mask)
        h, c = self._cell(torch.cat([emb, ctx], dim=-1), (state.h, state.c))
        logits = self._out(torch.cat([h, ctx], dim=-1))
        return DecoderOutput(logits=logits, state=DecoderState(h, c), attn=attn)
