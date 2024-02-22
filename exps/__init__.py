import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Callable

from omegaconf import OmegaConf, DictConfig
import torch
from torch import distributed as dist
from torch import multiprocessing as mp

import calf
from calf.utils.cuda import get_free_gpus, get_free_gpus_by_gid
from calf.utils.distributed import get_free_port, is_master
from calf.utils.log import init_logger, logger


def init_config(config_path: str = None,
                config_file: str = None,
                args: Dict = None) -> DictConfig:
    default_args = {"nproc": 0,
                    "gid": None,
                    "gpu_mem_threshold": 15278,
                    "seed": 25,
                    "threads": 16,
                    "verbose": True,
                    "log_to_file": False,
                    "checkpoint": False,
                    "use_available_gpu": False,
                    "log_master_only": True}
    config = OmegaConf.create()
    config.merge_with(default_args)
    if config_path and config_file:
        file = Path(config_path) / config_file
        if file.is_file():
            config.merge_with(OmegaConf.load(str(file)))
    if args is not None:
        config.merge_with(args)
    return config


def merge_config(config_path: str = None,
                 config_file: str = None,
                 cfg: DictConfig = None) -> DictConfig:
    cfg = OmegaConf.create() if cfg is None else cfg
    if config_path and config_file:
        file = Path(config_path) / config_file
        if file.is_file():
            cfg.merge_with(OmegaConf.load(str(file)))
    return cfg


def setup_os_environment() -> None:
    # disable parallelism to avoid deadlocks
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # disable warnings
    warnings.filterwarnings("ignore")
    # disable transformers warning
    # debug, info, warning, error, critical
    os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"
    # huggingface cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(calf.HUGGINGFACE / "hub")
    # # where assets created by downstream libraries will be cached locally
    # os.environ["HUGGINGFACE_ASSETS_CACHE"] = str(calf.HUGGINGFACE / "assets")
    # # only files that are already cached will be accessed
    # os.environ["HF_HUB_OFFLINE"] = '1'
    # # user access token to authenticate to the hub
    # os.environ["HUGGING_FACE_HUB_TOKEN"] = str(calf.HUGGINGFACE / "token")


def compete_for_gpus(cfg: DictConfig) -> List:
    # setup parallel environment
    if not cfg.nproc:
        gpus_to_use = []
        print(f"# [cpu]")
    else:
        if cfg.gid is not None:
            gpus_to_use = get_free_gpus_by_gid(cfg.gid)
        else:
            gpus_to_use = get_free_gpus(
                gpu_mem_threshold=cfg.gpu_mem_threshold,
                max_gpus=cfg.nproc) if cfg.nproc > 0 else []
            assert (
                    cfg.use_available_gpu or len(gpus_to_use) == cfg.nproc
            ), (f"{cfg.nproc} GPU(s) required, but only {len(gpus_to_use)} "
                f"available")
        print('\n'.join(
            [f"#{i} [gpu_id: {gid}, free_memory: {mem}M]" for i, (gid, mem) in enumerate(gpus_to_use)]
        ))
    gids = [g[0] for g in gpus_to_use]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(g) for g in gids])
    return gids


def run_experiment(
        experiment: str,
        command: str,
        config_path: str,
        config_file: str,
        callback: Callable = None,
        **kwargs
) -> None:
    # parse configuration
    args = locals()
    args.pop("callback")
    args.pop("kwargs")
    config_path = args.pop("config_path")
    config_file = args.pop("config_file")
    for k, v in kwargs.items():
        args[k] = v
    cfg = init_config(
        config_path=config_path,
        config_file=config_file,
        args=args
    )
    # setup os environment
    setup_os_environment()
    # get available gpus
    gpus = compete_for_gpus(cfg)
    # run experiment
    world_size = len(gpus)
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "tcp://127.0.0.1"
        os.environ["MASTER_PORT"] = get_free_port()
        mp.spawn(start, args=(world_size, callback, cfg), nprocs=world_size)
    else:
        start(0 if torch.cuda.is_available() else -1, world_size, callback, cfg)


def start(
        local_rank: int,
        world_size: int,
        fn: Callable,
        cfg: DictConfig
) -> None:
    # init dist
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method=f"{os.environ['MASTER_ADDR']}:"
                        f"{os.environ['MASTER_PORT']}",
            world_size=world_size,
            rank=local_rank
        )
    torch.cuda.set_device(local_rank)
    os.environ["RANK"] = os.environ["LOCAL_RANK"] = f"{local_rank}"
    cfg.local_rank = local_rank
    cfg.world_size = world_size
    # set calf device
    calf.device = f"cuda:{local_rank}" if world_size > 0 else "cpu"
    # # set output path
    # calf.OUTPUT = calf.OUTPUT / cfg.experiment
    # calf.OUTPUT.mkdir(parents=True, exist_ok=True)
    # cache path
    if not cfg.get("cache_path"):
        cache_path = calf.OUTPUT / "cache"
        cache_path.mkdir(parents=True, exist_ok=True)
        cfg.cache_path = str(cache_path)
    # checkpoint path
    if not cfg.get("checkpoint_path"):
        checkpoint_path = calf.OUTPUT / "checkpoint"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        cfg.checkpoint_path = str(checkpoint_path)
    # log path
    if not cfg.get("log_path"):
        log_path = calf.OUTPUT / "log"
        log_path.mkdir(parents=True, exist_ok=True)
        cfg.log_path = str(log_path)
    # update calf cfg
    calf.cfg.merge_with(cfg)
    # print only master node
    if not is_master() and cfg.log_master_only:
        sys.stdout = open(os.devnull, 'w')
    # log only master node
    log_file = str(Path(cfg.log_path) / f"{cfg.command}_{local_rank}.log")
    init_logger(
        logger=logger,
        log_file=log_file if cfg.log_to_file else None,
        mode='a' if cfg.checkpoint else 'w',
        verbose=cfg.verbose,
        non_master_level=logging.ERROR if cfg.log_master_only else logging.INFO
    )
    calf.logger = logger
    # callback
    if fn is not None:
        fn(cfg)