import sys
from pathlib import Path
import fire
from omegaconf import DictConfig
from exps import run_experiment, merge_config


def setup(cfg: DictConfig) -> None:
    from .setup.run import check_environment
    check_environment(cfg)


def corpus(cfg: DictConfig) -> None:
    from .corpus.run import load_corpus
    load_corpus(cfg)


def concept(cfg: DictConfig):
    from .concept.run import run_concept_lab
    run_concept_lab(cfg)


def syntax(cfg: DictConfig) -> None:
    from .syntax.run import run_syntax_labs
    run_syntax_labs(cfg)


def protoqa(cfg: DictConfig) -> None:
    from .protoqa.run import run_protoqa_lab
    run_protoqa_lab(cfg)


def commqa(cfg: DictConfig) -> None:
    from .commqa.run import run_commqa_lab
    run_commqa_lab(cfg)


def rep(cfg: DictConfig):
    from .representation import categorize, regression
    func = cfg.get("func", "all")
    if func == "categorize":
        categorize(cfg)
    elif func == "reg":
        regression(cfg)
    elif func == "all":
        categorize(cfg)
        regression(cfg)
    else:
        raise ValueError(f"func {func} not supported")


def start(cfg: DictConfig):
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    getattr(sys.modules[__name__], cfg.command)(cfg)


def main(command: str, config_path: str = None, config_file: str = None, **kwargs) -> None:
    run_experiment(
        experiment=Path(__file__).parent.stem,
        command=command,
        config_path=config_path if config_path else str(Path(__file__).parent),
        config_file=config_file if config_file else "config.yaml",
        callback=start,
        **kwargs
    )


if __name__ == "__main__":
    fire.Fire(main)
