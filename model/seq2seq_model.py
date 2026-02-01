# model/seq2seq_model.py
import torch
import torch.nn as nn

from data.batch import Batch
from model.lstm_encoder import LSTMEncoder
from model.lstm_decoder import LSTMDecoder
from model.encoder_output import EncoderOutput


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder: LSTMEncoder, decoder: LSTMDecoder, bos_id: int):
        super().__init__()
        self._enc = encoder
        self._dec = decoder
        self._bos_id = bos_id

    def forward(self, batch: Batch) -> torch.Tensor:
        enc_out = self._encode(batch)
        return self._decode_teacher_forcing(batch.tgt_ids, enc_out)

    def _encode(self, batch: Batch) -> EncoderOutput:
        return self._enc(batch.src_ids, batch.src_mask)

    def _decode_teacher_forcing(self, tgt_ids: torch.Tensor, enc_out: EncoderOutput) -> torch.Tensor:
        prev = self._decoder_inputs(tgt_ids)
        steps = prev.shape[1]
        state = self._init_state(prev)
        logits = self._alloc_logits(prev, steps)
        return self._loop_decode(prev, logits, state, enc_out)

    def _decoder_inputs(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        return tgt_ids[:, :-1]

    def _init_state(self, prev: torch.Tensor):
        bsz = prev.shape[0]
        return self._dec.init_state(bsz, prev.device)

    def _alloc_logits(self, prev: torch.Tensor, steps: int) -> torch.Tensor:
        bsz = prev.shape[0]
        vocab = self._dec._out.out_features 
        return torch.zeros((bsz, steps, vocab), device=prev.device)

    def _loop_decode(self, prev: torch.Tensor, logits: torch.Tensor, state, enc_out: EncoderOutput) -> torch.Tensor:
        for t in range(prev.shape[1]):
            out = self._dec.step(prev[:, t], state, enc_out.states, enc_out.mask)
            logits[:, t, :] = out.logits
            state = out.state
        return logits
