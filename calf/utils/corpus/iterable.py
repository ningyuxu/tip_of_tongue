from typing import List, Iterable
from .corpus import Corpus


class IterableCorpus(Corpus):

    corpus = "python_iterable"
    version = "v1"
    fields = ["text"]

    def __init__(self, values: Iterable = None) -> None:
        super().__init__(name=self.corpus)

        self.values = values

    def is_initialized(self) -> bool:
        return True

    def init(self) -> None:
        ...

    def reset(self) -> None:
        ...

    def load(self) -> Iterable[List]:
        for text in self.values:
            text = text.strip()
            if not text:
                continue
            else:
                yield [text]
