import json
from pathlib import Path
from typing import List, Dict

from omegaconf import DictConfig

from calf import CORPUS
from exps import merge_config


def all_hellaswag_dataset(cfg: DictConfig) -> List[Dict]:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    split = cfg.get("split", "dev")
    assert split in ["train", "dev", "test"]
    corpus_file = cfg.hellaswag_corpus_file[split]
    label_file = cfg.hellaswag_corpus_file[f"{split}_labels"]
    corpus_file = CORPUS / "hellaswag" / corpus_file
    label_file = CORPUS / "hellaswag" / label_file
    with open(corpus_file, mode='r', encoding="utf-8") as fp:
        json_list = list(fp)
    with open(label_file, mode='r', encoding="utf-8") as fp:
        label_list = [int(i) for i in list(fp)]
    dataset = []
    for json_str, label in zip(json_list, label_list):
        data = _parse_hellaswag_json_line(json_str, label)
        dataset.append(data)
    return dataset


def _parse_hellaswag_json_line(json_line: str, label: int) -> Dict:
    data = json.loads(json_line)
    data_id = data["id"]
    context = data["ctx"]
    choices = data["ending_options"]
    return {
        "data_id": data_id,
        "context": context,
        "question": '',
        "choices": choices,
        "answer_index": label
    }


