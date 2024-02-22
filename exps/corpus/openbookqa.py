import json
from pathlib import Path
from typing import List, Dict

from omegaconf import DictConfig

from calf import CORPUS
from exps import merge_config


def all_openbookqa_dataset(cfg: DictConfig) -> List[Dict]:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    split = cfg.get("split", "test")
    assert split in ["train", "dev", "test"]
    file = cfg.obqa_corpus_file[split]
    corpus_file = CORPUS / "openbook_qa" / file
    with open(corpus_file, mode='r', encoding="utf-8") as fp:
        json_list = list(fp)
    dataset = []
    for json_str in json_list:
        data = _parse_obqa_json_line(json_str)
        dataset.append(data)
    return dataset


def _parse_obqa_json_line(json_line: str) -> Dict:
    data = json.loads(json_line)
    data_id = data["id"]
    choices = {
        w["label"]: w["text"] for w in data["question"]["choices"]
    }
    question = data["question"]["stem"]
    answer_key = data.get("answerKey", '')
    if answer_key:
        answer_index = list(choices.keys()).index(answer_key)
    else:
        answer_index = -1
    return {
        "data_id": data_id,
        "context": '',
        "question": question,
        "choices": list(choices.values()),
        "answer_index": answer_index
    }

