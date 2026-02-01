import torch

from data.batch import Batch
from model.seq2seq_model import Seq2SeqModel
from training.loss_computer import LossComputer


class TrainStep:
    def __init__(self, model: Seq2SeqModel, loss_fn: LossComputer, optimizer: torch.optim.Optimizer):
        self._model = model
        self._loss_fn = loss_fn
        self._opt = optimizer

    def run(self, batch: Batch) -> float:
        self._model.train()
        self._opt.zero_grad()
        logits = self._model(batch)
        loss = self._loss_fn.compute(logits, batch.tgt_ids)
        loss.backward()
        self._opt.step()
        return float(loss.item())
