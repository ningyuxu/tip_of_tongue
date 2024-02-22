import os
import shutil
import gzip
import zipfile
import tarfile
import requests
import mmap
import struct
import pickle
import urllib
from pathlib import Path
from collections import defaultdict
from typing import Optional, Union, Any, Iterable, Tuple, List, Dict
import torch
from calf import cfg, logger
from calf.utils.log import progress_bar


def download(url: str, file: Optional[str] = None) -> Tuple[str, int]:
    if not file:
        file = cfg.cache_root / Path(urllib.parse.urlparse(url).path).name  # noqa
    try:
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024
        bar = progress_bar(logger, total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(file, mode='wb') as fp:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                fp.write(data)
        bar.close()
    except (KeyboardInterrupt, Exception) as error:
        if file.is_file():
            file.unlink(missing_ok=True)
        raise error
    return str(file), total_size_in_bytes


def extract(compressed_file: str,
            destination_folder: Optional[str] = None,
            clean: bool = False) -> str:
    assert Path(compressed_file).is_file(), f"{compressed_file} does not exist"
    path = Path(compressed_file).parent
    extracted = ""
    # zip format
    if zipfile.is_zipfile(compressed_file):
        with zipfile.ZipFile(compressed_file) as f:
            extracted = str(path / f.infolist()[0].filename)
            f.extractall(str(path))
    # tar format
    elif tarfile.is_tarfile(compressed_file):
        with tarfile.open(compressed_file) as f:
            extracted = str(path / f.getnames()[0])
            f.extractall(str(path))
    # gz format
    elif compressed_file.endswith(".gz"):
        extracted = compressed_file[:-3]
        with gzip.open(compressed_file) as fgz:
            with open(extracted, 'wb') as f:
                shutil.copyfileobj(fgz, f)
    else:
        raise ValueError(f"{compressed_file} not supported file type")
    if destination_folder:
        (path / extracted).rename(destination_folder)
    if clean:
        Path(compressed_file).unlink()
    return extracted


def binarize(data: Union[List[str], Dict[str, Iterable]],
             bin_file: Optional[str] = None,
             merge: bool = False) -> Tuple[str, Dict]:
    # the binarized file is organized as:
    # `data`: pickled objects
    # `meta`: a dict containing the pointers to each dataitem
    # `index`: fixed size integers representing the storage positions of the metadata
    start, meta = 0, defaultdict(list)
    with open(bin_file, mode='wb') as fw:
        if merge:  # in this case, data should be a list of binarized files
            for file in data:
                if not Path(file).is_file():
                    raise RuntimeError(f"{file} not exist")
                mi = debinarize(file, meta=True)
                for key, val in mi.items():
                    val[:, 0] += start
                    meta[key].append(val)
                with open(file, mode='rb') as fi:
                    length = int(sum(val[:, 1].sum() for val in mi.values()))
                    fw.write(fi.read(length))
                start = start + length
            meta = {key: torch.cat(val) for key, val in meta.items()}
        else:
            for key, vals in data.items():
                for v in vals:
                    bv = pickle.dumps(v)
                    fw.write(bv)
                    meta[key].append((start, len(bv)))
                    start = start + len(bv)
            meta = {key: torch.tensor(val) for key, val in meta.items()}
        pickled = pickle.dumps(meta)
        # append the metadata to the end of the bin file
        fw.write(pickled)
        # record the positions of the metadata
        fw.write(struct.pack('LL', start, len(pickled)))
    return bin_file, meta


def debinarize(bin_file: str,
               pos_or_key: Union[Tuple[int, int], str] = (0, 0),
               meta: bool = False) -> Union[Any, Iterable[Any]]:
    with open(bin_file, mode='rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        if meta or isinstance(pos_or_key, str):
            length = len(struct.pack('LL', 0, 0))
            mm.seek(-length, os.SEEK_END)
            offset, length = struct.unpack('LL', mm.read(length))
            mm.seek(offset)
            if meta:
                return pickle.loads(mm.read(length))
            # fetch by key
            objs, meta = [], pickle.loads(mm.read(length))[pos_or_key]
            for offset, length in meta.tolist():
                mm.seek(offset)
                objs.append(pickle.loads(mm.read(length)))
            return objs
        # fetch by positions
        offset, length = pos_or_key
        mm.seek(offset)
        return pickle.loads(mm.read(length))
