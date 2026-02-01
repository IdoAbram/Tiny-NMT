# inference/greedy_decoder.py
import torch

from model.seq2seq_model import Seq2SeqModel


class GreedyDecoder:
    def __init__(self, model: Seq2SeqModel, bos_id: int, eos_id: int, max_len: int = 50):
        self._model = model
        self._bos = bos_id
        self._eos = eos_id
        self._max_len = max_len

    @torch.no_grad()
    def translate(self, src_ids: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        self._model.eval()
        enc_out = self._model._enc(src_ids, src_mask)
        return self._decode(enc_out, src_ids.device)

    def _decode(self, enc_out, device: torch.device) -> torch.Tensor:
        bsz = enc_out.states.shape[0]
        state = self._model._dec.init_state(bsz, device)
        prev = self._start_tokens(bsz, device)
        return self._loop(prev, state, enc_out)

    def _start_tokens(self, bsz: int, device: torch.device) -> torch.Tensor:
        return torch.full((bsz,), self._bos, dtype=torch.long, device=device)

    def _loop(self, prev: torch.Tensor, state, enc_out) -> torch.Tensor:
        out_ids = [prev]
        for _ in range(self._max_len):
            step = self._model._dec.step(prev, state, enc_out.states, enc_out.mask)
            prev = self._pick(step.logits)
            state = step.state
            out_ids.append(prev)
            if self._all_eos(prev):
                break
        return torch.stack(out_ids, dim=1)

    def _pick(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)

    def _all_eos(self, ids: torch.Tensor) -> bool:
        return bool(torch.all(ids == self._eos).item())
