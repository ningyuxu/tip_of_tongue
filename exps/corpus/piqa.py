import json
from pathlib import Path
from typing import List, Dict

from omegaconf import DictConfig

from calf import CORPUS
from exps import merge_config


def all_piqa_dataset(cfg: DictConfig) -> List[Dict]:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    split = cfg.get("split", "dev")
    assert split in ["train", "dev", "test"]
    corpus_file = cfg.piqa_corpus_file[split]
    label_file = cfg.piqa_corpus_file[f"{split}_labels"]
    corpus_file = CORPUS / "piqa" / corpus_file
    label_file = CORPUS / "piqa" / label_file
    with open(corpus_file, mode='r', encoding="utf-8") as fp:
        json_list = list(fp)
    with open(label_file, mode='r', encoding="utf-8") as fp:
        label_list = [int(i) for i in list(fp)]
    dataset = []
    for json_str, label in zip(json_list, label_list):
        data = _parse_piqa_json_line(json_str, label)
        dataset.append(data)
    return dataset


def _parse_piqa_json_line(json_line: str, label: int) -> Dict:
    data = json.loads(json_line)
    data_id = data["id"]
    question = data["goal"]
    choices = list(data.values())[2:]
    return {
        "data_id": data_id,
        "context": '',
        "question": question,
        "choices": choices,
        "answer_index": label
    }

