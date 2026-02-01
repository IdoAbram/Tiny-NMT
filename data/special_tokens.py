from data.constants import Constants


class SpecialTokens:
    def pad(self) -> str:
        return Constants.PAD

    def bos(self) -> str:
        return Constants.BOS

    def eos(self) -> str:
        return Constants.EOS

    def unk(self) -> str:
        return Constants.UNK

    def all(self) -> list[str]:
        return [self.pad(), self.bos(), self.eos(), self.unk()]
