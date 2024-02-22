import json
from pathlib import Path
from typing import List, Dict

from omegaconf import DictConfig

from calf import CORPUS
from exps import merge_config


def all_protoqa_dataset(cfg: DictConfig) -> List[Dict]:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    split = cfg.get("split", "dev")
    assert split in ["train", "dev", "dev_scraped"]
    file = cfg.protoqa_corpus_file[split]
    path = CORPUS / "protoqa"
    corpus_file = path / file
    with open(corpus_file, mode='r', encoding="utf-8") as fp:
        json_list = list(fp)
    dataset = []
    for json_str in json_list:
        data = _parse_protoqa_json_line(json_str)
        dataset.append(data)
    return dataset


def _parse_protoqa_json_line(json_line: str) -> Dict:
    data = json.loads(json_line)
    data_id = data["metadata"]["id"]
    question = data["question"]["normalized"]
    return {
        "data_id": data_id,
        "question": question,
    }
