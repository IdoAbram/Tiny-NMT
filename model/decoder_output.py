import torch
from typing import Optional
from model.decoder_state import DecoderState


class DecoderOutput:
    def __init__(self, logits: torch.Tensor, state: DecoderState, attn: Optional[torch.Tensor] = None):
        self.logits = logits
        self.state = state
        self.attn = attn
