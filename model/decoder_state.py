import torch


class DecoderState:
    def __init__(self, h: torch.Tensor, c: torch.Tensor):
        self.h = h
        self.c = c
