from data.constants import Constants


class SequenceRules:
    def __init__(self, special_ids: dict[str, int]):
        self._pad = special_ids["pad"]
        self._bos = special_ids["bos"]
        self._eos = special_ids["eos"]

    def add_source_markers(self, ids: list[int]) -> list[int]:
        return ids + [self._eos]

    def add_target_markers(self, ids: list[int]) -> list[int]:
        return [self._bos] + ids + [self._eos]
