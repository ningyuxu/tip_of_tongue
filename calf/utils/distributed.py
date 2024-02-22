import functools
from typing import Any, List
import torch.distributed as dist


def wait(fn) -> Any:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        value = None
        if is_master():
            value = fn(*args, **kwargs)
        if is_dist():
            dist.barrier()
            value = gather(value)[0]
        return value
    return wrapper


def gather(obj: Any) -> List[Any]:
    objs = [None] * dist.get_world_size()
    dist.all_gather_object(objs, obj)
    return objs


def reduce(obj: Any, reduction: str = "sum") -> Any:
    objs = gather(obj)
    if reduction == "sum":
        return functools.reduce(lambda x, y: x + y, objs)
    elif reduction == "mean":
        return functools.reduce(lambda x, y: x + y, objs) / len(objs)
    elif reduction == "min":
        return min(objs)
    elif reduction == "max":
        return max(objs)
    else:
        raise NotImplementedError(f"Unsupported reduction {reduction}")


def is_dist():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist():
        return 0
    return dist.get_rank()


def is_master():
    return get_rank() == 0


def get_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))
    port = str(s.getsockname()[1])
    s.close()
    return port
