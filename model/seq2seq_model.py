# model/seq2seq_model.py
import torch
import torch.nn as nn

from data.batch import Batch
from model.lstm_encoder import LSTMEncoder
from model.lstm_decoder import LSTMDecoder
from model.encoder_output import EncoderOutput
from model.decoder_state import DecoderState


class Seq2SeqModel(nn.Module):
    """
    Seq2Seq model with:
    - LSTMEncoder producing (states, final_hidden, final_cell)
    - LSTMDecoder with attention

    Improvement:
    - Initialize decoder state (h0, c0) from encoder final state (projected).
    """

    def __init__(self, encoder: LSTMEncoder, decoder: LSTMDecoder, bos_id: int):
        super().__init__()
        self._enc = encoder
        self._dec = decoder
        self._bos_id = bos_id

        # Encoder final state dim: H * (2 if bidirectional else 1)
        enc_h = self._enc._cfg.hidden_dim
        enc_dirs = 2 if self._enc._cfg.bidirectional else 1
        enc_final_dim = enc_h * enc_dirs

        # Decoder hidden dim:
        dec_h = self._dec._cell.hidden_size

        # Project encoder final -> decoder initial state
        self._h0_proj = nn.Linear(enc_final_dim, dec_h)
        self._c0_proj = nn.Linear(enc_final_dim, dec_h)

    def forward(self, batch: Batch) -> torch.Tensor:
        enc_out = self._encode(batch)
        return self._decode_teacher_forcing(batch.tgt_ids, enc_out)

    def _encode(self, batch: Batch) -> EncoderOutput:
        return self._enc(batch.src_ids, batch.src_mask)

    def _decode_teacher_forcing(self, tgt_ids: torch.Tensor, enc_out: EncoderOutput) -> torch.Tensor:
        prev = self._decoder_inputs(tgt_ids)
        steps = prev.shape[1]
        state = self._init_state_from_encoder(prev, enc_out)
        logits = self._alloc_logits(prev, steps)
        return self._loop_decode(prev, logits, state, enc_out)

    def _decoder_inputs(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        # teacher forcing input tokens
        return tgt_ids[:, :-1]

    def _alloc_logits(self, prev: torch.Tensor, steps: int) -> torch.Tensor:
        bsz = prev.shape[0]
        vocab = self._dec._out.out_features
        return torch.zeros((bsz, steps, vocab), device=prev.device)

    def _loop_decode(self, prev: torch.Tensor, logits: torch.Tensor, state: DecoderState, enc_out: EncoderOutput) -> torch.Tensor:
        for t in range(prev.shape[1]):
            out = self._dec.step(prev[:, t], state, enc_out.states, enc_out.mask)
            logits[:, t, :] = out.logits
            state = out.state
        return logits

    # -----------------------------
    # NEW: decoder init from encoder
    # -----------------------------

    def init_decoder_state(self, enc_out: EncoderOutput) -> DecoderState:
        """
        Public helper for inference decoders (greedy/beam).
        Creates (h0,c0) from encoder final state.
        """
        enc_h = self._last_layer_concat_dirs(enc_out.final_hidden)  # [B, H*D]
        enc_c = self._last_layer_concat_dirs(enc_out.final_cell)    # [B, H*D]

        h0 = torch.tanh(self._h0_proj(enc_h))
        c0 = torch.tanh(self._c0_proj(enc_c))
        return DecoderState(h0, c0)

    def _init_state_from_encoder(self, prev: torch.Tensor, enc_out: EncoderOutput) -> DecoderState:
        # prev is only used for device/batch size consistency
        # We compute state from encoder output (preferred)
        return self.init_decoder_state(enc_out)

    def _last_layer_concat_dirs(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [L*D, B, H] from nn.LSTM
        returns: [B, H*D] for the last layer
        """
        num_layers = self._enc._cfg.num_layers
        bidir = self._enc._cfg.bidirectional
        D = 2 if bidir else 1

        start = (num_layers - 1) * D
        end = start + D
        last = x[start:end]  # [D, B, H]

        if D == 1:
            return last[0]  # [B, H]
        else:
            # concatenate forward + backward
            return torch.cat([last[0], last[1]], dim=-1)  # [B, 2H]
