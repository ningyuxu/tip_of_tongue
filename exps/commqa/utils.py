from pathlib import Path
from typing import Tuple

from omegaconf import DictConfig

from calf import OUTPUT
from calf.modules import get_model, get_tokenizer


# -------------------- File Functions --------------------
def get_commqa_root() -> Path:
    path = OUTPUT / "commqa"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_commqa_result_file(task: str) -> Path:
    path = get_commqa_root()
    file = f"commqa_{task}_results.json"
    return path / file


# -------------------- Model Functions --------------------
def get_model_and_tokenizer(model_cfg: DictConfig, model_dtype: str) -> Tuple:
    revision = model_cfg.get("revision", None)
    trust_remote_code = model_cfg.get("trust_remote_code", False)
    model = get_model(
        model_name=model_cfg.name,
        revision=revision,
        model_dtype=model_dtype,
        n_context=model_cfg.n_context,
        trust_remote_code=trust_remote_code
    )
    tokenizer = get_tokenizer(
        model_name=model_cfg.name,
        revision=revision,
        trust_remote_code=trust_remote_code
    )
    return model, tokenizer
