import json
from pathlib import Path
from typing import List, Dict

from omegaconf import DictConfig
from wordfreq import word_frequency, zipf_frequency

from calf import CORPUS
from exps import init_config, merge_config


def _unify_concrete_pos(pos: str) -> str:
    mapper = {
        "0": "X",
        "Adjective": "ADJ",
        "Adverb": "ADV",
        "Article": "DET",
        "Conjunction": "CCONJ",
        "Determiner": "DET",
        "Ex": "ADV",
        "Interjection": "INTJ",
        "Letter": "PRON",
        "Name": "NOUN",
        "Not": "ADV",
        "Noun": "NOUN",
        "Number": "NUM",
        "Preposition": "ADP",
        "Pronoun": "PRON",
        "To": "ADP",
        "Unclassified": "X",
        "Verb": "VERB",
        "#N/A": "X",
    }
    return mapper.get(pos, "X")


def _parse_data_line(line: str) -> Dict:
    values = line.strip().split('\t')
    word = values[0].strip().replace(' ', '_')
    bigram = values[1].strip()
    conc_m = values[2].strip()
    percent_known = values[6].strip()
    subtlex = values[7].strip()
    pos = _unify_concrete_pos(values[8].strip())
    return {
        "word": word,
        "bigram": True if bigram == '1' else False,
        "conc_m": float(conc_m),
        "percent_known": float(percent_known),
        "subtlex": int(subtlex),
        "pos": pos,
        "word_f": word_frequency(
            word.replace('_', ' '),
            lang="en"
        ),
        "zipf_f": zipf_frequency(
            word.replace('_', ' '),
            lang="en"
        ),
    }


def all_concrete_words() -> List[Dict]:
    cfg = init_config(
        config_path=str(Path(__file__).parent),
        config_file="config.yaml"
    )
    corpus_file = CORPUS / "concreteness" / cfg.concrete_corpus_file
    with open(corpus_file, mode='r', encoding="utf-8") as fp:
        words = []
        for i, line in enumerate(fp):
            if i == 0:  # pass head
                continue
            data = _parse_data_line(line)
            if data:
                words.append(data)
    return words


def concrete_lookup_table(cfg: DictConfig, reload: bool = False) -> Dict:
    cfg = merge_config(
        config_path=str(Path(__file__).parent),
        config_file="config.yaml",
        cfg=cfg
    )
    table_file = Path(cfg.cache_path) / cfg.concrete_lookup_table_file
    if reload or not table_file.is_file():
        words = all_concrete_words()
        table = {}
        for data in words:
            table[data["word"]] = data["conc_m"]
        with open(table_file, mode='w', encoding="utf-8") as fp:
            json.dump(table, fp)
    else:
        with open(table_file, mode='r', encoding="utf-8") as fp:
            table = json.load(fp)
    return table


def concrete_filter(
        words: List[Dict],
        bigram: bool = None,
        min_conc_m: float = None,
        max_conc_m: float = None,
        percent_known: float = None,
        min_subtlex: int = None,
        max_subtlex: int = None,
        pos: List = None,
        min_word_f: float = None,
        max_word_f: float = None,
        min_zipf_f: float = None,
        max_zipf_f: float = None,
) -> List[Dict]:
    filter_words = []
    for word in words:
        if bigram is not None and bigram ^ word["bigram"]:
            continue
        if min_conc_m is not None and word["conc_m"] < min_conc_m:
            continue
        if max_conc_m is not None and word["conc_m"] >= max_conc_m:
            continue
        if percent_known is not None and word["percent_known"] < percent_known:
            continue
        if min_subtlex is not None and word["subtlex"] < min_subtlex:
            continue
        if max_subtlex is not None and word["subtlex"] >= max_subtlex:
            continue
        if pos is not None and word["pos"] not in pos:
            continue
        if min_word_f is not None and word["word_f"] < min_word_f:
            continue
        if max_word_f is not None and word["word_f"] >= max_word_f:
            continue
        if min_zipf_f is not None and word["zipf_f"] < min_zipf_f:
            continue
        if max_zipf_f is not None and word["zipf_f"] >= max_zipf_f:
            continue
        filter_words.append(word)
    return filter_words
