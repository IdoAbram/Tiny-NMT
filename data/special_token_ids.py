from data.constants import Constants
from data.vocabulary import Vocabulary


class SpecialTokenIds:
    def __init__(self, vocab: Vocabulary):
        self._vocab = vocab

    def pad_id(self) -> int:
        return self._vocab.token_to_id(Constants.PAD)

    def bos_id(self) -> int:
        return self._vocab.token_to_id(Constants.BOS)

    def eos_id(self) -> int:
        return self._vocab.token_to_id(Constants.EOS)

    def unk_id(self) -> int:
        return self._vocab.token_to_id(Constants.UNK)

    def as_dict(self) -> dict[str, int]:
        return {
            "pad": self.pad_id(),
            "bos": self.bos_id(),
            "eos": self.eos_id(),
            "unk": self.unk_id(),
        }
