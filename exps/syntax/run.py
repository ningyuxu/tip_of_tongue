import sys

from omegaconf import DictConfig


def run_syntax_labs(cfg: DictConfig) -> None:
    lab_name = cfg.get("lab", "syntaxgym")
    getattr(sys.modules[__name__], lab_name)(cfg)


def syntaxgym(cfg: DictConfig) -> None:
    from .sg_eval_analysis import calc_sg_surprisal, calc_sg_score
    func = cfg.get("func", "surprisal")
    if func == "surprisal":
        calc_sg_surprisal(cfg)
    elif func == "score":
        calc_sg_score(cfg)
    else:
        raise ValueError(f"func {func} not supported")


def blimp(cfg: DictConfig) -> None:
    from .blimp_analysis import calc_blimp_probability, calc_blimp_score
    func = cfg.get("func", "probability")
    if func == "probability":
        calc_blimp_probability(cfg)
    elif func == "score":
        calc_blimp_score(cfg)
    else:
        raise ValueError(f"func {func} not supported")
