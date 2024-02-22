import json
from pathlib import Path
from typing import List, Dict

from omegaconf import DictConfig

from calf import CORPUS
from exps import merge_config


def all_boolq_dataset(cfg: DictConfig) -> List[Dict]:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    split = cfg.get("split", "dev")
    assert split in ["train", "dev", "test"]
    corpus_file = cfg.boolq_corpus_file[split]
    corpus_file = CORPUS / "boolq" / corpus_file
    with open(corpus_file, mode='r', encoding="utf-8") as fp:
        json_list = list(fp)
    dataset = []
    for i, json_str in enumerate(json_list):
        data = _parse_boolq_json_line(i, json_str)
        dataset.append(data)
    return dataset


def _parse_boolq_json_line(seqid: int, json_line: str) -> Dict:
    data_id = seqid
    data = json.loads(json_line)
    context = data["passage"]
    question = data["question"]
    choices = ["Yes", "No"]
    answer = data["answer"]
    return {
        "data_id": data_id,
        "context": context,
        "question": question,
        "choices": choices,
        "answer_index": 1 - int(answer)
    }