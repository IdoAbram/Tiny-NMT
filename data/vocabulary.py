from collections import Counter
from data.constants import Constants
from data.special_tokens import SpecialTokens
from data.tokenizer import Tokenizer

MAX_VOCAB_SIZE = 20000 


class Vocabulary:
    def __init__(self, tokenizer: Tokenizer, specials: SpecialTokens):
        self._tok = tokenizer
        self._sp = specials
        self._token2id: dict[str, int] = {}
        self._id2token: dict[int, str] = {}
        self._is_built = False
        self._init_specials()

    def build(self, texts: list[str]) -> None:
        counts = self._count_tokens(texts)

        # ממיינים לפי שכיחות (גבוה → נמוך)
        sorted_tokens = sorted(
            counts.items(), key=lambda x: x[1], reverse=True
        )

        for token, freq in sorted_tokens:
            if freq < Constants.MIN_FREQ:
                continue

            if len(self._token2id) >= MAX_VOCAB_SIZE:
                break

            self._add(token)

        self._is_built = True

    def encode(self, text: str) -> list[int]:
        self._require_built()
        tokens = self._tok.tokenize(text)
        return [self.token_to_id(t) for t in tokens]

    def decode(self, ids: list[int], skip_specials: bool = True) -> str:
        tokens = [self.id_to_token(i) for i in ids]
        if skip_specials:
            tokens = [t for t in tokens if not self._is_special(t)]
        return " ".join(tokens).strip()

    def token_to_id(self, token: str) -> int:
        return self._token2id.get(token, self._token2id[self._sp.unk()])

    def id_to_token(self, idx: int) -> str:
        return self._id2token.get(idx, self._sp.unk())

    def __len__(self) -> int:
        return len(self._token2id)

    def special_id(self, token: str) -> int:
        return self._token2id[token]

    def _init_specials(self) -> None:
        for t in self._sp.all():
            self._add(t)

    def _add(self, token: str) -> None:
        if token in self._token2id:
            return
        idx = len(self._token2id)
        self._token2id[token] = idx
        self._id2token[idx] = token

    def _count_tokens(self, texts: list[str]) -> Counter:
        c = Counter()
        for text in texts:
            c.update(self._tok.tokenize(text))
        return c

    def _require_built(self) -> None:
        if not self._is_built:
            raise RuntimeError("Vocabulary not built yet. Call build(texts) first.")

    def _is_special(self, token: str) -> bool:
        return token in set(self._sp.all())
