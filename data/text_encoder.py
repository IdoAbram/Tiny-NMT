from data.special_tokens import SpecialTokens
from data.vocabulary import Vocabulary


class TextEncoder:
    def __init__(self, vocab: Vocabulary, specials: SpecialTokens):
        self._vocab = vocab
        self._sp = specials

    def encode_source(self, text: str) -> list[int]:
        # Problem 1: content-only. (Later you may append EOS here.)
        return self._vocab.encode(text)

    def encode_target(self, text: str) -> list[int]:
        # Problem 1: content-only. (Later you add BOS/EOS in Problem 3.)
        return self._vocab.encode(text)

    def get_special_ids(self) -> dict[str, int]:
        return {
            "pad": self._vocab.special_id(self._sp.pad()),
            "bos": self._vocab.special_id(self._sp.bos()),
            "eos": self._vocab.special_id(self._sp.eos()),
            "unk": self._vocab.special_id(self._sp.unk()),
        }
