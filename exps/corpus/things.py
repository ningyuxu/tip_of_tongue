import json
from pathlib import Path
from typing import List, Dict

from wordfreq import word_frequency, zipf_frequency

from calf import CORPUS
from exps import init_config, merge_config


def _unify_things_pos(pos: str) -> str:
    mapper = {
        "Noun": "NOUN",
        "Name": "NOUN",
        "Verb": "VERB",
        "Adjective": "ADJ",
        "Adverb": "ADV",
        "#N/A": 'X',  # unknown
    }
    return mapper.get(pos, 'X')


def _parse_line_for_concrete(head: List, line: str) -> Dict:
    values = [v.strip('"') for v in line.strip().split('\t')]
    head_values = {h: v for h, v in zip(head, values)}
    word = head_values["Word"].strip().replace(' ', '_')
    bigram = True if head_values["Bigram"] == '1' else False
    conc_m = float(head_values["Concreteness (M)"])
    s_percent_known = head_values["Percent_known"]
    percent_known = float(s_percent_known) if s_percent_known else 0
    s_subtlex = head_values["SUBTLEX freq"]
    subtlex = int(s_subtlex) if s_subtlex else 0
    pos = _unify_things_pos(head_values["Dominant Part of Speech"])
    return {
        "word": word,
        "bigram": bigram,
        "conc_m": conc_m,
        "percent_known": percent_known,
        "subtlex": subtlex,
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


def _parse_line_for_wordnet(head: List, line: str) -> Dict:
    values = [v.strip('"') for v in line.strip().split('\t')]
    head_values = {h: v for h, v in zip(head, values)}
    synset_name = head_values["Wordnet ID4"]
    word = head_values["Word"].strip().replace(' ', '_')
    pos = _unify_things_pos(head_values["Dominant Part of Speech"])
    description = head_values[
        "Definition (from WordNet, Google, or Wikipedia)"
    ]
    synonyms = head_values["WordNet Synonyms"].strip().split(',')
    synonyms = [s.strip().replace(' ', '_') for s in synonyms]
    return {
        "synset": synset_name,
        "word": word,
        "pos": pos,
        "description": description,
        "lemmas": synonyms,
        "examples": []
    }


def _parse_line_for_category(head: List, line: str) -> Dict:
    values = [v.strip('"') for v in line.strip().split('\t')]
    head_values = {h: v for h, v in zip(head, values)}
    unique_id = head_values["uniqueID"]
    synset_name = head_values["Wordnet ID4"]
    word = head_values["Word"].strip().replace(' ', '_')
    pos = _unify_things_pos(head_values["Dominant Part of Speech"])
    description = head_values[
        "Definition (from WordNet, Google, or Wikipedia)"
    ]
    synonyms = head_values["WordNet Synonyms"].strip().split(',')
    synonyms = [s.strip().replace(' ', '_') for s in synonyms]
    return {
        "uniqueID": unique_id,
        "synset": synset_name,
        "word": word,
        "pos": pos,
        "description": description,
        "lemmas": synonyms,
        "examples": []
    }


def all_things_dataset(fmt: str = "wordnet") -> List[Dict]:
    """
    To retrieve all things data in certain format. Format can be `wordnet`,
    `concrete`.
    """
    cfg = init_config(
        config_path=str(Path(__file__).parent),
        config_file="config.yaml"
    )
    concepts_file = CORPUS / "things" / cfg.things_corpus_file
    with open(concepts_file, mode='r', encoding="utf-8") as fp:
        things = []
        for i, line in enumerate(fp):
            if i == 0:
                head = [h.strip('"') for h in line.strip().split('\t')]
                continue
            if fmt == "wordnet":
                thing = _parse_line_for_wordnet(head, line)
            elif fmt == "concrete":
                thing = _parse_line_for_concrete(head, line)
            elif fmt == "category":
                thing = _parse_line_for_category(head, line)
            else:
                raise ValueError(f"Format {fmt} not supported")
            things.append(thing)
    return things


def specific_category_dataset(category: str, in_category: bool = True) -> List:
    cfg = init_config(
        config_path=str(Path(__file__).parent),
        config_file="config.yaml"
    )
    things = all_things_dataset(fmt="category")
    category_file = CORPUS / "things" / cfg.thingsplus_category_file
    words_of_category = []
    with open(category_file, mode='r', encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            if i == 0:
                continue
            values = [v.strip('"') for v in line.strip().split(',')]
            if values[3] == category:
                words_of_category.append(values[1].replace(' ', '_'))
    if in_category:
        things_of_category = [
            t for t in things if t["uniqueID"] in words_of_category
        ]
    else:
        things_of_category = [
            t for t in things if t["uniqueID"] not in words_of_category
        ]
    return things_of_category


def word_synset_to_uniqueid_table(cfg) -> None:
    cfg = merge_config(
        config_path=str(Path(__file__).parent),
        config_file="config.yaml",
        cfg=cfg
    )
    concepts_file = CORPUS / "things" / cfg.things_corpus_file
    with open(concepts_file, mode='r', encoding="utf-8") as fp:
        table = {}
        for i, line in enumerate(fp):
            if i == 0:
                head = [h.strip('"') for h in line.strip().split('\t')]
                continue
            values = [v.strip('"') for v in line.strip().split('\t')]
            head_values = {h: v for h, v in zip(head, values)}
            synset = head_values["Wordnet ID4"]
            word = head_values["Word"].strip().replace(' ', '_')
            unique_id = head_values["uniqueID"]
            table[f"{word}_{synset}"] = unique_id
    table_file = Path(cfg.cache_path) / "things_search_uniqueid.json"
    with open(table_file, mode='w', encoding="utf-8") as fp:
        json.dump(table, fp)
