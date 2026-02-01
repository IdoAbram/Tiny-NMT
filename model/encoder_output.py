import torch
from typing import Optional


class EncoderOutput:
    def __init__(
        self,
        states: torch.Tensor,
        final_hidden: torch.Tensor,
        final_cell: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        self.states = states
        self.final_hidden = final_hidden
        self.final_cell = final_cell
        self.mask = mask
