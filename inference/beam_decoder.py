# inference/beam_decoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from model.seq2seq_model import Seq2SeqModel
from model.encoder_output import EncoderOutput
from model.decoder_state import DecoderState


@dataclass
class _Hyp:
    ids: List[int]
    score: float
    state: DecoderState
    ended: bool


class BeamSearchDecoder:
    """
    Beam search per-sample (robust & simple).
    Uses length penalty to reduce preference for too-short outputs.
    """

    def __init__(
        self,
        model: Seq2SeqModel,
        bos_id: int,
        eos_id: int,
        max_len: int = 50,
        beam_size: int = 5,
        len_penalty_alpha: float = 0.6,
    ):
        self._model = model
        self._bos = int(bos_id)
        self._eos = int(eos_id)
        self._max_len = int(max_len)
        self._beam = int(beam_size)
        self._alpha = float(len_penalty_alpha)

        assert self._beam >= 1

    @torch.no_grad()
    def translate(self, src_ids: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        self._model.eval()
        enc_out = self._model._enc(src_ids, src_mask)

        device = src_ids.device
        B = enc_out.states.shape[0]
        outputs: List[torch.Tensor] = []

        for i in range(B):
            enc_i = self._slice_enc_out(enc_out, i)
            best = self._decode_one(enc_i, device)
            outputs.append(best)

        # pad outputs to same length
        maxT = max(t.size(0) for t in outputs)
        padded = []
        for t in outputs:
            if t.size(0) < maxT:
                pad = torch.full((maxT - t.size(0),), self._eos, dtype=torch.long, device=device)
                padded.append(torch.cat([t, pad], dim=0))
            else:
                padded.append(t)

        return torch.stack(padded, dim=0)

    def _slice_enc_out(self, enc_out: EncoderOutput, i: int) -> EncoderOutput:
        # states: [B,S,E], mask: [B,S], final_hidden/cell: [L*D,B,H]
        return EncoderOutput(
            states=enc_out.states[i : i + 1],
            final_hidden=enc_out.final_hidden[:, i : i + 1, :],
            final_cell=enc_out.final_cell[:, i : i + 1, :],
            mask=enc_out.mask[i : i + 1] if enc_out.mask is not None else None,
        )

    def _len_penalty(self, length: int) -> float:
        if self._alpha <= 0.0:
            return 1.0
        # Common: ((5+len)/6)^alpha
        return ((5.0 + float(length)) / 6.0) ** self._alpha

    def _decode_one(self, enc_out: EncoderOutput, device: torch.device) -> torch.Tensor:
        # NEW: init decoder from encoder final state (instead of zeros)
        state0 = self._model.init_decoder_state(enc_out)

        beam: List[_Hyp] = [_Hyp(ids=[self._bos], score=0.0, state=state0, ended=False)]
        finished: List[_Hyp] = []

        for _ in range(self._max_len):
            if len(finished) >= self._beam:
                break

            candidates: List[_Hyp] = []

            for hyp in beam:
                if hyp.ended:
                    candidates.append(hyp)
                    continue

                prev_id = torch.tensor([hyp.ids[-1]], dtype=torch.long, device=device)  # [1]
                step = self._model._dec.step(prev_id, hyp.state, enc_out.states, enc_out.mask)

                logits = step.logits.squeeze(0)  # [V]
                log_probs = torch.log_softmax(logits, dim=-1)

                topk_logp, topk_ids = torch.topk(log_probs, k=self._beam)

                for k in range(self._beam):
                    tok = int(topk_ids[k].item())
                    tok_lp = float(topk_logp[k].item())

                    new_ids = hyp.ids + [tok]
                    new_score = hyp.score + tok_lp
                    ended = (tok == self._eos)

                    # clone state tensors to avoid aliasing between hypotheses
                    new_state = DecoderState(
                        h=step.state.h.clone(),
                        c=step.state.c.clone(),
                    )

                    candidates.append(_Hyp(ids=new_ids, score=new_score, state=new_state, ended=ended))

            # rank by length-penalized score
            candidates.sort(key=lambda h: h.score / self._len_penalty(len(h.ids)), reverse=True)

            # build next beam
            next_beam: List[_Hyp] = []
            for h in candidates:
                if h.ended:
                    finished.append(h)
                else:
                    next_beam.append(h)
                if len(next_beam) >= self._beam:
                    break

            beam = next_beam
            if len(beam) == 0:
                break

        pool = finished if len(finished) > 0 else beam
        pool.sort(key=lambda h: h.score / self._len_penalty(len(h.ids)), reverse=True)
        best = pool[0]
        return torch.tensor(best.ids, dtype=torch.long, device=device)
