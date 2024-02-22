import tempfile
import shutil
from pathlib import Path
from typing import List, Iterable
from contextlib import contextmanager
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from calf import logger, CACHE
from ..common import INF, get_signature
from ..file import binarize, debinarize
from ..log import progress_bar
from ..distributed import is_master, is_dist
from ..corpus import Corpus
from ..transform import Transform


class Dataset(torch.utils.data.Dataset):  # noqa

    def __init__(self,
                 corpus: Corpus,
                 transform: Transform,
                 max_len: int = None,
                 cache: bool = True,
                 cache_path: str = None,
                 cache_file: str = None,
                 reload: bool = False,
                 chunk_size: int = 1000) -> None:
        super().__init__()

        self.corpus = corpus
        self.transform = transform
        self.max_len = max_len or INF
        self.cache = cache
        self.cache_file = self.get_cache_file(cache_path, cache_file)
        self.reload = reload
        self.chunk_size = chunk_size
        self._data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.cache:  # meta data
            return debinarize(str(self.cache_file), self.data[index])
        else:  # actural data
            return self.data[index]

    @property
    def data(self):  # either actural data (not cached) or meta data (cached)
        if self._data is None:
            self._data = self.load()
        return self._data

    def get_cache_file(self, path: str, file: str) -> Path:
        path = Path(path) if path else CACHE
        path.mkdir(parents=True, exist_ok=True)
        file = file if file else get_signature({"corpus": self.corpus.name,
                                                "transform": self.transform.name})
        return path / file

    def load(self) -> List:
        if self.cache:
            if not self.reload and self.cache_file.is_file():  # already cached
                data = debinarize(str(self.cache_file), meta=True)["data"]
            else:  # cache and return
                logger.info(f"Seeking to cache the data to {str(self.cache_file)} first")
                data = []
                if is_master():
                    data = self.transform.load(corpus=self.corpus)
                    with self.binarize(data) as chunks, mp.Pool(32) as pool:
                        results = [pool.apply_async(self.numericalize, chunk) for chunk in chunks]
                        data = binarize([r.get() for r in results],
                                        str(self.cache_file), merge=True)[1]["data"]
                if is_dist():
                    dist.barrier()
                if not is_master():
                    data = debinarize(str(self.cache_file), meta=True)["data"]
        else:
            data = self.transform.load(corpus=self.corpus)
            data = [item for item in self.transform(data) if len(item) < self.max_len]
        return data

    @contextmanager
    def binarize(self, data: Iterable) -> Iterable:
        ftmp = Path(tempfile.mkdtemp())
        fs = ftmp / "dataset"
        fb = ftmp / self.cache_file.name
        data = binarize({"data": progress_bar(logger, iterator=data)}, str(fs))[1]["data"]
        try:
            yield ((data[s:s + self.chunk_size], fs, f"{fb}.{i}", self.max_len)
                   for i, s in enumerate(range(0, len(data), self.chunk_size)))
        finally:
            shutil.rmtree(ftmp)

    def numericalize(self, data, fs, fb, max_len) -> str:
        data = self.transform(debinarize(fs, item) for item in data)
        data = [item for item in data if len(item) < max_len]
        return binarize({"data": data}, fb)[0]
