import json
from pathlib import Path
from typing import Dict

from omegaconf import DictConfig

from calf import CORPUS
from exps import merge_config


def all_blimp_dataset(cfg: DictConfig) -> Dict:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    file_pattern = cfg.blimp_corpus_file_pattern
    corpus_path = CORPUS / "blimp"
    corpus_files = list(corpus_path.glob(file_pattern))
    blimp_topics = {}
    for file in corpus_files:
        topic = file.stem
        with open(file, mode='r', encoding="utf-8") as fp:
            json_list = list(fp)
        dataset = []
        for i, json_str in enumerate(json_list):
            data = _parse_blimp_json_line(i, json_str)
            dataset.append(data)
        blimp_topics[topic] = dataset
    return blimp_topics


def _parse_blimp_json_line(seqid: int, json_line: str) -> Dict:
    data_id = seqid
    data = json.loads(json_line)
    sentence_good = data["sentence_good"]
    sentence_bad = data["sentence_bad"]
    field = data["field"]
    linguistics_term = data["linguistics_term"]
    UID = data["UID"]
    simple_LM_method = data["simple_LM_method"]
    one_prefix_method = data["one_prefix_method"]
    two_prefix_method = data["two_prefix_method"]
    lexically_identical = data["lexically_identical"]
    pairID = data["pairID"]
    return {
        "seqid": data_id,
        "sentence_good": sentence_good,
        "sentence_bad": sentence_bad,
        "field": field,
        "linguistics_term": linguistics_term,
        "UID": UID,
        "simple_LM_method": simple_LM_method,
        "one_prefix_method": one_prefix_method,
        "two_prefix_method": two_prefix_method,
        "lexically_identical": lexically_identical,
        "pairID": pairID,
    }