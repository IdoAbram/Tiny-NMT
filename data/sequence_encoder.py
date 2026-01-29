from data.vocabulary import Vocabulary
from data.sequence_rules import SequenceRules


class SequenceEncoder:
    def __init__(self, vocab: Vocabulary, rules: SequenceRules):
        self._vocab = vocab
        self._rules = rules

    def encode_source(self, text: str) -> list[int]:
        ids = self._vocab.encode(text)
        return self._rules.add_source_markers(ids)

    def encode_target(self, text: str) -> list[int]:
        ids = self._vocab.encode(text)
        return self._rules.add_target_markers(ids)
