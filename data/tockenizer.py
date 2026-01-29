import re
from data.constants import Constants


class Tokenizer:
    def normalize(self, text: str) -> str:
        t = text
        if Constants.STRIP:
            t = t.strip()
        t = re.sub(r"\s+", " ", t)
        if Constants.LOWERCASE:
            t = t.lower()
        return t

    def tokenize(self, text: str) -> list[str]:
        t = self.normalize(text)
        return t.split(" ") if t else []
