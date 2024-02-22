import os
import io
import shutil
import contextlib
import tempfile
import zipfile
from typing import Literal, Union
import numpy as np


class _TempMMap:

    def __init__(self, data_source, mmap_mode):
        # why to use ``NamedTemporaryFile`` without automatic removal:
        # https://github.com/numpy/numpy/issues/3143
        self.cbuf = tempfile.NamedTemporaryFile(delete=False)
        try:
            with contextlib.closing(data_source):
                shutil.copyfileobj(data_source, self.cbuf)
        except Exception:
            self.close()
            raise Exception
        else:
            self.close(_delete=False)
        self.mmap_mode = mmap_mode

    def open(self):
        return np.load(self.cbuf.name, mmap_mode=self.mmap_mode)

    def close(self, _delete=True):
        if self.cbuf is not None:
            self.cbuf.close()
        if _delete and self.cbuf is not None:
            try:
                os.remove(self.cbuf.name)
            except FileNotFoundError:
                self.cbuf = None
            except Exception:
                raise Exception(f"Error removing temp file {self.cbuf.name}")
            else:
                self.cbuf = None

    def __enter__(self):
        return self.open()

    def __exit__(self, _1, _2, _3):
        self.close()


class NpzMMap:
    def __init__(self, npzfile) -> None:
        self.npzfile = npzfile
        with np.load(self.npzfile) as zdata:
            self.npzkeys = set(zdata)
        self._zfile = zipfile.ZipFile(self.npzfile)

    def close(self):
        if self._zfile is not None:
            self._zfile.close()

    def mmap(self, key: str, mmap_mode: str = 'r'):
        if key not in self.npzkeys:
            raise KeyError(f"key '{key}' not in npzfile '{self.npzfile}'")
        if not mmap_mode:
            raise ValueError("mmap_mode must not be empty")
        if mmap_mode != 'r':
            raise NotImplementedError
        if key not in self._zfile.namelist():
            key += ".npy"
        assert key in self._zfile.namelist(), str(key)
        return _TempMMap(self._zfile.open(key), mmap_mode)

    def __enter__(self):
        return self

    def __exit__(self, _1, _2, _3):
        self.close()


class IncrementalNpzWriter:
    """
    Write data to npz file incrementally rather than compute all and write once, as in ``np.save``.
    This class can be used with ``contextlib.closing`` to ensure closed after usage.
    """
    def __init__(self, tofile: str, mode: Literal['x', 'w', 'a'] = 'x'):
        """
        Args:
            tofile: the ``npz`` file to write
            mode: must be one of {'x', 'w', 'a'}.
                See https://docs.python.org/3/library/zipfile.html for detail
        """
        assert mode in "xwa", str(mode)
        self.tofile = zipfile.ZipFile(file=tofile, mode=mode, compression=zipfile.ZIP_DEFLATED)

    def write(self,
              key: str,
              data: Union[np.ndarray, bytes],
              is_npy_data: bool = True) -> None:
        """
        Args:
            key: the name of data to write
            data: the data
            is_npy_data: if ``True``, ".npy" will be appended to ``key``, and ``data`` will be
                serialized by ``np.save``; otherwise, ``key`` will be treated as is, and ``data``
                will be treated as binary data
        """
        if key in self.tofile.namelist():
            raise KeyError(f"Duplicate key '{key}' already exists in '{self.tofile.filename}'")
        self.update(key, data, is_npy_data=is_npy_data)

    def update(self, key: str, data: Union[np.ndarray, bytes], is_npy_data: bool = True) -> None:
        kwargs = {"mode": 'w', "force_zip64": True}
        if is_npy_data:
            key += ".npy"
            with io.BytesIO() as cbuf:
                np.save(cbuf, data)  # noqa
                cbuf.seek(0)
                with self.tofile.open(key, **kwargs) as outfile:
                    shutil.copyfileobj(cbuf, outfile)
        else:
            with self.tofile.open(key, **kwargs) as outfile:
                outfile.write(data)

    def close(self):
        if self.tofile is not None:
            self.tofile.close()
            self.tofile = None
