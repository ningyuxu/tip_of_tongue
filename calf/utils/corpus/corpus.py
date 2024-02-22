import shutil
from typing import Iterable
from torch import distributed as dist
from calf import logger, CORPUS, CACHE
from calf.utils.file import download, extract
from ..distributed import is_master, is_dist


class Corpus:

    corpus = ''
    version = ''
    fields = []
    corpus_path = CORPUS / f"{corpus}_{version}"
    corpus_filename = ''
    url = ''

    def __init__(self, reload: bool = False, **kwargs) -> None:
        if reload:
            self.reset()
        self.name = '_'.join([str(v) for v in kwargs.values()])

    def __len__(self):
        return sum(1 for _ in self.data)

    @property
    def data(self) -> Iterable:
        if not self.is_initialized():
            self.init()
        return self.load()

    @classmethod
    def is_initialized(cls) -> bool:
        return cls.corpus_path.is_dir() and any(cls.corpus_path.iterdir())

    def reset(self) -> None:
        if is_master():
            shutil.rmtree(self.corpus_path, ignore_errors=True)
        if is_dist():
            dist.barrier()

    def init(self) -> None:
        if is_master():
            # reset corpus
            self.reset()
            # download corpus
            corpus_file = str(CACHE / self.corpus_filename)
            logger.info(f"Downloading {self.corpus} corpus ...")
            download(url=self.url, file=corpus_file)
            logger.info(f"Done.")
            logger.info(f"Extracting {self.corpus} corpus ...")
            extract(compressed_file=corpus_file,
                    destination_folder=str(self.corpus_path),
                    clean=True)
            logger.info(f"Done")
        if is_dist():
            dist.barrier()

    def load(self) -> Iterable:
        raise NotImplementedError
