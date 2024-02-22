from typing import List, Dict, Tuple

from omegaconf import DictConfig


def get_support_template(cfg: DictConfig) -> str:
    task = cfg.embedding.task
    if task == "w2d":
        support_template = cfg.w2d_s_template
    elif task == "cw2d":
        support_template = cfg.cw2d_s_template
    elif task == "d2w":
        support_template = cfg.d2w_s_template
    elif task == "ld2w":
        support_template = cfg.ld2w_s_template
    elif task == "w2w":
        support_template = cfg.w2w_s_template
    else:
        raise ValueError(f"Task {task} not recognized.")
    return support_template


def get_query_template(cfg: DictConfig) -> str:
    task = cfg.embedding.task
    if task == "w2d":
        query_template = cfg.w2d_q_template
    elif task == "cw2d":
        query_template = cfg.cw2d_q_template
    elif task == "d2w":
        query_template = cfg.d2w_q_template
    elif task == "ld2w":
        query_template = cfg.ld2w_q_template
    elif task == "w2w":
        query_template = cfg.w2w_q_template
    else:
        raise ValueError(f"Task {task} not recognized.")
    return query_template


def get_desc_tag(cfg: DictConfig) -> str:
    task = cfg.get("task", "d2w")
    if task == "ld2w":
        desc_tag = cfg.t_nl_desc
    else:
        desc_tag = cfg.t_desc
    return desc_tag


def fill_in_template(cfg: DictConfig, data: Dict, template: str) -> str:  # noqa
    example = data["example"]
    word = data["word"]
    description = data["description"]
    result = template.replace("$example", example)
    result = result.replace("$word", word)
    result = result.replace("$description", description)
    return result


def prepare_support_prompt(cfg: DictConfig, support_dataset: List) -> str:
    template = get_support_template(cfg)
    prompt = ''
    for data in support_dataset:
        demo = fill_in_template(cfg, data, template)
        prompt = f"{prompt}\n{demo}" if prompt else f"{demo}"
    return prompt


def prepare_one_prompt(
        cfg: DictConfig,
        support_prompt: str,
        query_data: Dict
) -> Tuple[str, str]:
    template = get_query_template(cfg)
    q_prompt = fill_in_template(cfg, query_data, template)
    prompt = f"{support_prompt}\n{q_prompt}" if support_prompt else q_prompt
    desc_tag = get_desc_tag(cfg)
    return prompt, desc_tag