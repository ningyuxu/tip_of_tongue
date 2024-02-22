import shutil
from typing import Iterable
from calf import logger, CORPUS, CACHE
from .. import enumeration as enum
from ..file import download, extract
from .corpus import Corpus


class WikiText103Corpus(Corpus):

    corpus = "wikitext103"
    version = "v1"
    fields = ["paragraph"]
    corpus_path = CORPUS / f"{corpus}_{version}"
    corpus_filename = f"{corpus}_{version}.zip"

    def __init__(self,
                 url: str =
                 "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip",
                 split: str = enum.Split.TRAIN,
                 reload: bool = False) -> None:
        super().__init__(reload=reload, name=self.corpus, split=split)
        self.url = url
        self.split = split

    def is_initialized(self) -> bool:
        return self.corpus_path.is_dir() and any(self.corpus_path.iterdir())

    def init(self) -> None:
        # reset corpus
        self.reset()
        # download corpus
        wikitext103_file = CACHE / self.corpus_filename
        logger.info(f"Downloading {self.corpus} corpus ...")
        download(url=self.url, file=wikitext103_file)
        logger.info(f"Done.")
        logger.info(f"Extracting {self.corpus} corpus ...")
        extract(compressed_file=wikitext103_file, destination_folder=self.corpus_path, clean=True)
        logger.info(f"Done")

    def reset(self) -> None:
        shutil.rmtree(self.corpus_path, ignore_errors=True)

    def load(self) -> Iterable[str]:
        with open(self.get_file(), mode='r', encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('='):
                    continue
                else:
                    yield [line]

    def get_file(self) -> str:
        split = "valid" if self.split == enum.Split.DEV else self.split
        return str(self.corpus_path / f"wiki.{split}.tokens")
