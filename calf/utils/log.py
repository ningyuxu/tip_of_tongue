import os
import logging
from logging import Logger, Formatter, Handler, FileHandler, StreamHandler
from tqdm import tqdm
from typing import Iterable, Optional, List, Dict, Union
import torch.distributed as dist
from .distributed import is_master


def get_logger(name: Optional[str] = None) -> Logger:
    logger = logging.getLogger(name)
    if name is None:
        logging.basicConfig(
            format="[%(asctime)s %(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[TqdmHandler()]
        )
    return logger


def init_logger(logger: Logger,
                log_file: Optional[str] = None,
                level: int = logging.INFO,
                non_master_level: int = logging.ERROR,
                mode: str = 'w',
                handlers: Optional[Iterable[Handler]] = None,
                verbose: bool = True) -> Logger:
    if not handlers:
        if log_file:
            os.makedirs(os.path.dirname(log_file) or './', exist_ok=True)
            logger.addHandler(FileHandler(log_file, mode))
    for handler in logger.handlers:
        handler.setFormatter(ColoredFormatter(colored=not isinstance(handler, FileHandler)))
    if verbose:
        logger.setLevel(level if is_master() else non_master_level)
    return logger


def print_log(msg: str, logger: Optional[Logger] = None, level: int = logging.INFO) -> None:
    if logger is None:
        print(msg)
    elif isinstance(logger, Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    else:
        raise TypeError(
            f"Logger should be either a logging.Logger object, 'silent' or None, but got {type(logger)}."
        )


def log_line(logger: Optional[Logger] = None, lens: int = 81, level: int = logging.INFO) -> None:
    msg = '-' * lens
    print_log(msg=msg, logger=logger, level=level)


def log_message(msg: str, logger: Optional[Logger] = None, level: int = logging.INFO) -> None:
    lines = msg.split('\n')
    for line in lines:
        if line.strip():
            print_log(msg=line, logger=logger, level=level)


def log_table(table: Union[Dict, List[Dict]],
              columns: Optional[List] = None,
              logger: Optional[Logger] = None,
              level: int = logging.INFO) -> None:
    # parse table head
    if isinstance(table, Dict):
        table = [table]
    if not columns:
        columns = list(table[0].keys() if table else [])
    p_list = [columns]  # 1st row = header

    # parse table line
    for item in table:
        p_list.append([str(item[col] or '') for col in columns])

    # format table
    # maximun size of the col for each element
    col_size = [max(map(len, col)) for col in zip(*p_list)]
    # insert seperating line before every line, and extra one for ending
    for i in range(0, len(p_list) + 1)[::-1]:
        p_list.insert(i, ['-' * i for i in col_size])
    # two format for each content line and each seperating line
    format_edg = "---".join(["{{:<{}}}".format(i) for i in col_size])
    format_str = " | ".join(["{{:<{}}}".format(i) for i in col_size])
    format_sep = "-+-".join(["{{:<{}}}".format(i) for i in col_size])

    # print table
    print_log(format_edg.format(*p_list[0]), logger, level=level)
    for item in p_list[1:-1]:
        if item[0][0] == '-':
            print_log(format_sep.format(*item), logger, level=level)
        else:
            print_log(format_str.format(*item), logger, level=level)
    print_log(format_edg.format(*p_list[-1]), logger, level=level)


def progress_bar(logger: Logger,
                 iterator: Iterable = None,
                 total: int = None,
                 ncols: Optional[int] = None,
                 bar_format: Optional[str] =
                 "{l_bar}{bar:20}| {n_fmt}/{total_fmt} {elapsed}<{remaining}, {rate_fmt}{postfix}",
                 leave: bool = False,
                 **kwargs) -> tqdm:
    return tqdm(
        iterator,
        total=total,
        ncols=ncols,
        bar_format=bar_format,
        ascii=False,
        disable=(not (logger.level == logging.INFO and is_main_process())),
        leave=leave,
        **kwargs
    )


class TqdmHandler(StreamHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class ColoredFormatter(Formatter):

    BLACK = "\033[30m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    GREY = "\033[37m"
    RESET = "\033[0m"

    COLORS = {
        logging.ERROR: RED,
        logging.WARNING: YELLOW,
        logging.INFO: GREEN,
        logging.DEBUG: GREY,
        logging.NOTSET: BLACK
    }

    def __init__(self, colored=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colored = colored

    def format(self, record):
        fmt = "[%(asctime)s %(levelname)s] %(message)s"
        if self.colored:
            fmt = f"{self.COLORS[record.levelno]}[%(asctime)s %(levelname)s]" \
                  f"{self.RESET} %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        return Formatter(fmt=fmt, datefmt=datefmt).format(record)


def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    else:
        return dist.get_rank() == 0


logger = get_logger()
