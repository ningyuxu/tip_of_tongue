from typing import Iterable, Dict
from calf import device
from calf.utils.corpus import Corpus


class Transform:
    """
    Usually, we need to fewshot things below:
    1) load dataset
    2) prepare input for model using loaded dataset
        - preprocess: filter data, convert datatype ...
        - transform: tokenize text, convert labels to ids ...
        - compose: padding to same length, to gpu ...
    """

    fields = []

    def __init__(self, name: str):
        self.name = name
        self.training = False

    def __call__(self, dataset: Iterable[Dict]) -> Iterable[Dict]:
        return self.transform(dataset)

    def train(self, training=True):
        self.training = training

    def eval(self):
        self.train(False)

    def load(self, corpus: Corpus) -> Iterable[Dict]:
        raise NotImplementedError

    def transform(self, dataset: Iterable[Dict]) -> Iterable[Dict]:
        raise NotImplementedError

    def collate(self, batch: Iterable[Dict]) -> Dict:
        batch = self.compose(batch)
        return {k: v.to(device=device) if hasattr(v, "to") else v for k, v in batch.items()}

    def compose(self, batch: Iterable[Dict]) -> Dict:
        raise NotImplementedError
