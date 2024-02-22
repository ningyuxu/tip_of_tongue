import json
from pathlib import Path
from typing import List, Dict

from calf import CORPUS
from exps import init_config


def all_revdic_dataset() -> List[Dict]:
    cfg = init_config(str(Path(__file__).parent), "config.yaml")
    file = cfg.revdic_corpus_file
    corpus_file = CORPUS / "reverse_dictionary" / file
    dataset = []
    with open(corpus_file, mode='r', encoding="utf-8") as fp:
        json_data = json.load(fp)
        for data in json_data:
            dataset.append(_parse_revdic_data(data))
    return dataset


def _parse_revdic_data(data: Dict) -> Dict:
    word = data["word"]
    description = data["definitions"]
    return {
        "synset": '',
        "word": word,
        "pos": '',
        "description": description,
        "lemmas": [word],
        "examples": []
    }
