import sys

from omegaconf import DictConfig


def run_commqa_lab(cfg: DictConfig) -> None:
    lab_name = cfg.get("lab", "natural")
    getattr(sys.modules[__name__], lab_name)(cfg)


def natural(cfg: DictConfig) -> None:
    from .natural_commqa import calc_commqa_probability, calc_commqa_accuracy
    func = cfg.get("func", "probability")
    if func == "probability":
        calc_commqa_probability(cfg)
    elif func == "accuracy":
        calc_commqa_accuracy(cfg)
    else:
        raise ValueError(f"func {func} not supported")


def concept(cfg: DictConfig) -> None:
    from .concept_commqa import calc_commqa_probability, calc_commqa_accuracy
    func = cfg.get("func", "probability")
    if func == "probability":
        calc_commqa_probability(cfg)
    elif func == "accuracy":
        calc_commqa_accuracy(cfg)
    else:
        raise ValueError(f"func {func} not supported")