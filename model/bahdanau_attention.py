import torch
import torch.nn as nn
from model.attention_config import AttentionConfig


class BahdanauAttention(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self._w_enc = nn.Linear(cfg.enc_dim, cfg.attn_dim, bias=False)
        self._w_dec = nn.Linear(cfg.dec_dim, cfg.attn_dim, bias=False)
        self._v = nn.Linear(cfg.attn_dim, 1, bias=False)

    def forward(self, dec_h: torch.Tensor, enc_states: torch.Tensor,
                src_mask: torch.Tensor):
        scores = self._scores(dec_h, enc_states)
        scores = self._apply_mask(scores, src_mask)
        attn = torch.softmax(scores, dim=1)
        ctx = self._context(attn, enc_states)
        return ctx, attn

    def _scores(self, dec_h: torch.Tensor, enc_states: torch.Tensor) -> torch.Tensor:
        e = self._w_enc(enc_states)
        d = self._w_dec(dec_h).unsqueeze(1)
        return self._v(torch.tanh(e + d)).squeeze(-1)

    def _apply_mask(self, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.tensor(-1e9, device=scores.device, dtype=scores.dtype)
        return scores.masked_fill(~mask, neg_inf)

    def _context(self, attn: torch.Tensor, enc_states: torch.Tensor) -> torch.Tensor:
        w = attn.unsqueeze(-1)
        return (w * enc_states).sum(dim=1)
