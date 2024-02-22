import sys
from omegaconf import DictConfig


def run_concept_lab(cfg: DictConfig) -> None:
    lab_name = cfg.get("lab", "inference")
    getattr(sys.modules[__name__], lab_name)(cfg)


def inference(cfg: DictConfig) -> None:
    from .concept_inference import (
        run_embed_concept,
        run_clone_concept,
        run_concept_baseline
    )
    from .output_results import (
        show_generate_results,
        show_exact_match_results
    )
    func = cfg.get("func", "embed_concept")
    if func == "embed_concept":
        run_embed_concept(cfg)
    elif func == "gener_results":
        show_generate_results(cfg)
    elif func == "match_results":
        show_exact_match_results(cfg)
    elif func == "clone_concept":
        run_clone_concept(cfg)
    elif func == "baseline":
        run_concept_baseline(cfg)
    else:
        raise ValueError(f"Functionality {func} not supported")


def influence(cfg: DictConfig) -> None:
    from .influence_factors import (
        run_influ_description,
        run_influ_concept,
        show_concept_influ_results,
        show_desc_influ_results,
    )
    func = cfg.get("func", "description")
    if func == "description":
        run_influ_description(cfg)
    elif func == "desc_influ_result":
        show_desc_influ_results(cfg)
    elif func == "concept":
        run_influ_concept(cfg)
    elif func == "concept_influ_result":
        show_concept_influ_results(cfg)
    else:
        raise ValueError(f"Functionality {func} not supported")