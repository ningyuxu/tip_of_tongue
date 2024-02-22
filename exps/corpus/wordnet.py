import json
from pathlib import Path
from typing import List, Dict

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

from calf import CORPUS, logger
from calf.utils.log import progress_bar
from exps import init_config
from .things import all_things_dataset


def _unify_wordnet_pos(pos: str) -> str:
    mapper = {
        'n': "NOUN",
        'v': "VERB",
        'a': "ADJ",
        'r': "ADV",
        's': "ADJ",
        'c': "CCONJ",
        'p': "ADP",  # preposition, postposition, etc.
        'x': "X",  # particle, classifier, exclamative, etc.
        'u': "X",  # unknown
    }
    return mapper.get(pos, 'X')


def _wordnet_dataset_file(cache_path: str, cache_file: str) -> Path:
    file_stem = Path(cache_file).stem
    file_ext = Path(cache_file).suffix
    file = f"{file_stem}"
    file = f"{file}_all{file_ext}"
    return Path(cache_path) / file


def _synset_hyponyms(synset: Synset, depth: int = 0) -> List:
    hyponyms = [synset]
    if depth > 0:
        for s in synset.hyponyms():
            hyponyms.extend(_synset_hyponyms(s, depth - 1))
    return hyponyms


def format_wordnet_synset(synset: Synset) -> Dict:
    synset_name = synset.name()
    word = synset.name().split('.')[0]
    pos = _unify_wordnet_pos(synset.pos())
    description = synset.definition()
    lemmas = synset.lemma_names()
    lemmas = [word] + lemmas if word not in lemmas else lemmas
    examples = synset.examples()
    return {
        "synset": synset_name,
        "word": word,
        "pos": pos,
        "description": description,
        "lemmas": lemmas,
        "examples": examples,
    }


def wordnet_synset_data(synset_name: str) -> Dict:
    synset = wn.synset(synset_name)
    return format_wordnet_synset(synset)


def wordnet_word_dataset(word: str) -> List:
    synsets = wn.synsets(word.replace(' ', '_'))
    dataset = []
    for synset in synsets:
        dataset.append(format_wordnet_synset(synset))
    return dataset


def wordnet_all_dataset(cache_path: str, reload: bool = False) -> List:
    cfg = init_config(
        config_path=str(Path(__file__).parent),
        config_file="config.yaml"
    )
    dataset_file = _wordnet_dataset_file(cache_path, cfg.wordnet_dataset_file)
    if reload or not dataset_file.is_file():
        dataset = []
        for synset in progress_bar(
                logger=logger,
                iterator=wn.all_synsets(),
                desc="Loading wordnet corpus ..."
        ):
            dataset.append(format_wordnet_synset(synset))
        with open(dataset_file, mode='w', encoding="utf-8") as fp:
            json.dump(dataset, fp)
    else:
        with open(dataset_file, mode='r', encoding="utf-8") as fp:
            dataset = json.load(fp)
    return dataset


def wordnet_tree_dataset(root_synset: str, tree_depth: int) -> List:
    synsets = _synset_hyponyms(wn.synset(root_synset), depth=tree_depth)
    dataset = []
    for synset in synsets:
        dataset.append(format_wordnet_synset(synset))
    return dataset


def wordnet_bc5000_dataset() -> List:
    from xml.etree import ElementTree as et
    import nltk
    from nltk.corpus import WordNetCorpusReader
    nltk.data.path.append(CORPUS)
    cfg = init_config(config_path=str(Path(__file__).parent), config_file="config.yaml")
    dataset = []
    wn2 = WordNetCorpusReader(
        root=str(CORPUS / cfg.wordnet20_dict_path),
        omw_reader=nltk.data.find(cfg.wordnet20_dict_path)
    )
    words_file = CORPUS / "bc5000" / cfg.bc5000_corpus_file
    with open(words_file, mode='r', encoding="utf-8") as fp:
        for line in fp:
            xml_root = et.fromstring(line)
            word = xml_root.find(
                "SYNONYM/LITERAL"
            ).text.replace(' ', '_')
            synset_id = xml_root.find("ID").text
            offset = int(synset_id.split('-')[1])
            pos = synset_id.split('-')[2]
            synset = wn2.synset_from_pos_and_offset(pos, offset)
            data = format_wordnet_synset(synset)
            data["word"] = word
            dataset.append(data)
    return dataset


def wordnet_thing_dataset() -> List:
    dataset = []
    for data in all_things_dataset(fmt="wordnet"):
        if data["synset"]:
            wn_data = format_wordnet_synset(wn.synset(data["synset"]))
            wn_data["word"] = data["word"]
            dataset.append(wn_data)
    return dataset