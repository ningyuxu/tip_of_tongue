import sys

from omegaconf import DictConfig


def run_protoqa_lab(cfg: DictConfig) -> None:
    lab_name = cfg.get("lab", "sample_answers")
    getattr(sys.modules[__name__], lab_name)(cfg)


def sample_answers(cfg: DictConfig) -> None:
    from .protoqa_labs import sample_protoqa_answers
    sample_protoqa_answers(cfg)
