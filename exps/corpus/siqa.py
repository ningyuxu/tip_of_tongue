import json
from pathlib import Path
from typing import List, Dict

from omegaconf import DictConfig

from calf import CORPUS
from exps import merge_config


def all_siqa_dataset(cfg: DictConfig) -> List[Dict]:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    split = cfg.get("split", "dev")
    assert split in ["train", "dev", "test"]
    corpus_file = cfg.siqa_corpus_file[split]
    label_file = cfg.siqa_corpus_file[f"{split}_labels"]
    corpus_file = CORPUS / "siqa" / corpus_file
    label_file = CORPUS / "siqa" / label_file
    with open(corpus_file, mode='r', encoding="utf-8") as fp:
        json_list = list(fp)
    with open(label_file, mode='r', encoding="utf-8") as fp:
        label_list = [int(i) for i in list(fp)]
    dataset = []
    for i, (json_str, label) in enumerate(zip(json_list, label_list)):
        data = _parse_siqa_json_line(i, json_str, label)
        dataset.append(data)
    return dataset


def _parse_siqa_json_line(seqid: int, json_line: str, label: int) -> Dict:
    data_id = seqid
    data = json.loads(json_line)
    context = data["context"]
    question = data["question"]
    choices = list(data.values())[2:]
    return {
        "data_id": data_id,
        "context": context,
        "question": question,
        "choices": choices,
        "answer_index": label - 1
    }


