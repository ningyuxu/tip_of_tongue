from typing import Tuple
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from calf.utils.distributed import is_dist
from ..corpus import Corpus
from ..transform import Transform
from .dataset import Dataset


def build_dataloader(corpus: Corpus,
                     transform: Transform,
                     max_len: int = None,
                     cache: bool = True,
                     reload: bool = False,
                     cache_path: str = None,
                     cache_file: str = None,
                     chunk_size: int = 1000,
                     training: bool = False,
                     batch_size: int = 1,
                     num_workers: int = 0,
                     pim_memory: bool = None,
                     drop_last: bool = False) -> Tuple[DataLoader, Dataset, DistributedSampler]:
    dataset = Dataset(
        corpus=corpus,
        transform=transform,
        max_len=max_len,
        cache=cache,
        reload=reload,
        cache_path=cache_path,
        cache_file=cache_file,
        chunk_size=chunk_size,
    )
    sampler = DistributedSampler(dataset,
                                 num_replicas=dist.get_world_size(),
                                 rank=dist.get_rank()) if is_dist() else None
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=pim_memory,
        num_workers=num_workers,
        shuffle=(sampler is None) and training,
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=transform.collate
    )
    return dataloader, dataset, sampler


__all__ = ["Dataset",
           "build_dataloader"]
