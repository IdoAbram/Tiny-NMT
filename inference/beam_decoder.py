# inference/beam_decoder.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch

from model.seq2seq_model import Seq2SeqModel
from model.decoder_state import DecoderState


@dataclass
class _Hyp:
    ids: List[int]
    score: float
    state: DecoderState
    ended: bool


class BeamSearchDecoder:
    """
    Beam Search decoding for your Seq2SeqModel (LSTM + Bahdanau attention).

    Notes:
    - Runs beam search per-sample (loops over batch dimension). Simple & robust.
    - Uses length penalty (optional) to avoid preferring too-short sequences.
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
        assert beam_size >= 1
        self._model = model
        self._bos = int(bos_id)
        self._eos = int(eos_id)
        self._max_len = int(max_len)
        self._beam = int(beam_size)
        self._alpha = float(len_penalty_alpha)

    @torch.no_grad()
    def translate(self, src_ids: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_ids:  [B, S]
            src_mask: [B, S] (bool/int)
        Returns:
            out_ids:  [B, T] where T <= max_len+1 (includes BOS and generated tokens)
        """
        self._model.eval()
        enc_out = self._model._enc(src_ids, src_mask)

        device = src_ids.device
        B = enc_out.states.shape[0]
        outputs: List[torch.Tensor] = []

        for i in range(B):
            # slice single example
            states_i = enc_out.states[i : i + 1]  # [1, S, enc_dim]
            mask_i = enc_out.mask[i : i + 1]      # [1, S]
            best_ids = self._decode_one(states_i, mask_i, device)
            outputs.append(best_ids)

        # pad to same length across batch
        maxT = max(t.size(0) for t in outputs)
        padded = []
        for t in outputs:
            if t.size(0) < maxT:
                pad = torch.full((maxT - t.size(0),), self._eos, dtype=torch.long, device=device)
                padded.append(torch.cat([t, pad], dim=0))
            else:
                padded.append(t)
        return torch.stack(padded, dim=0)  # [B, maxT]

    def _len_penalty(self, length: int) -> float:
        # Common MT length penalty: lp = ((5+len)/6)^alpha
        if self._alpha <= 0:
            return 1.0
        return ((5.0 + float(length)) / 6.0) ** self._alpha

    def _decode_one(self, enc_states: torch.Tensor, src_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Beam search for a single sample.
        enc_states: [1, S, enc_dim]
        src_mask:   [1, S]
        returns: 1D tensor of token ids (includes BOS, ends with EOS usually)
        """
        # init decoder state for a single sample
        state0 = self._model._dec.init_state(1, device)

        # start with BOS
        init = _Hyp(ids=[self._bos], score=0.0, state=state0, ended=False)
        beam: List[_Hyp] = [init]
        finished: List[_Hyp] = []

        for _t in range(self._max_len):
            # if we already have enough finished hyps, we can stop early
            if len(finished) >= self._beam:
                break

            candidates: List[_Hyp] = []

            for hyp in beam:
                if hyp.ended:
                    # keep ended hypotheses as-is
                    candidates.append(hyp)
                    continue

                prev_id = torch.tensor([hyp.ids[-1]], dtype=torch.long, device=device)  # [1]
                step = self._model._dec.step(prev_id, hyp.state, enc_states, src_mask)
                logits = step.logits.squeeze(0)  # [V]

                # log-probs
                log_probs = torch.log_softmax(logits, dim=-1)  # [V]

                # take top K next tokens for this hypothesis
                topk_logp, topk_ids = torch.topk(log_probs, k=self._beam)

                for k in range(self._beam):
                    tok = int(topk_ids[k].item())
                    tok_lp = float(topk_logp[k].item())

                    new_ids = hyp.ids + [tok]
                    new_score = hyp.score + tok_lp
                    ended = (tok == self._eos)

                    # state is a DecoderState with tensors of shape [1, hidden]
                    # IMPORTANT: clone to avoid accidental aliasing between hyps
                    new_state = DecoderState(
                        h=step.state.h.clone(),
                        c=step.state.c.clone(),
                    )

                    candidates.append(_Hyp(ids=new_ids, score=new_score, state=new_state, ended=ended))

            # rank candidates by length-penalized score
            candidates.sort(key=lambda h: h.score / self._len_penalty(len(h.ids)), reverse=True)

            # next beam = best K candidates that are not finished
            beam = []
            for h in candidates:
                if h.ended:
                    finished.append(h)
                else:
                    beam.append(h)
                if len(beam) >= self._beam:
                    break

            # if no alive hyps remain, stop
            if len(beam) == 0:
                break

        # choose best from finished if any, else from beam
        pool = finished if len(finished) > 0 else beam
        pool.sort(key=lambda h: h.score / self._len_penalty(len(h.ids)), reverse=True)
        best = pool[0]

        return torch.tensor(best.ids, dtype=torch.long, device=device)
