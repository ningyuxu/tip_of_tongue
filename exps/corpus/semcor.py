import json
import random
from pathlib import Path
from typing import List, Dict

from omegaconf import DictConfig
from nltk.corpus import semcor
from nltk.tree.tree import Tree
from nltk.corpus.reader.wordnet import Lemma

from calf import logger
from calf.utils.log import progress_bar
from exps import merge_config


def _unify_semcor_pos(pos: str) -> str:
    mapper = {
        "JJ": "ADJ",
        "JJR": "ADJ",
        "JJS": "ADJ",
        "RB": "ADV",
        "RBR": "ADV",
        "RBS": "ADV",
        "NN": "NOUN",
        "NNS": "NOUN",
        "VB": "VERB",
        "VBD": "VERB",
        "VBG": "VERB",
        "VBN": "VERB",
        "VBP": "VERB",
        "VBZ": "VERB",
    }
    return mapper.get(pos, 'X')


def _get_semcor_dataset_file(cfg: DictConfig) -> Path:
    file_stem = Path(cfg.semcor_dataset_file).stem
    file_ext = Path(cfg.semcor_dataset_file).suffix
    file = f"{file_stem}_all{file_ext}"
    return Path(cfg.cache_path) / file


def _get_semcor_inverted_index_file(cfg: DictConfig) -> Path:
    return Path(cfg.cache_path) / cfg.semcor_inverted_index_file


def parse_sentence(sentence_tree: Tree) -> List[Dict]:
    parsed_chunks = []
    sentence = ' '.join([w for c in sentence_tree for w in c.leaves()])
    for chunk in sentence_tree:
        pos = _unify_semcor_pos(chunk.pos()[0][1])
        if pos == 'X':
            continue
        if isinstance(chunk.label(), Lemma):
            synset = chunk.label().synset()
            word = [w for w in chunk.leaves()]
            parsed_chunks.append({
                "synset": synset.name(),
                "word": word,
                "pos": pos,
                "examples": [sentence],
            })
    return parsed_chunks


def all_semcor_dataset(cfg: DictConfig, reload: bool = False) -> List[List[Dict]]:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    dataset_file = _get_semcor_dataset_file(cfg)
    if reload or not dataset_file.is_file():
        logger.info("Loading data from semcor corpus ...")
        sentences = [s for s in semcor.tagged_sents(tag="both")]
        dataset = []
        for tagged_sent in progress_bar(
                logger=logger,
                iterator=sentences,
                desc="Parsing data in semcor corpus ..."
        ):
            dataset.append(parse_sentence(tagged_sent))
        with open(dataset_file, mode='w', encoding="utf-8") as fp:
            json.dump(dataset, fp)
    else:
        with open(dataset_file, mode='r', encoding="utf-8") as fp:
            dataset = json.load(fp)
    return dataset


def sample_semcor_dataset(cfg: DictConfig, n_samples: int, same_sentence_first: bool) -> List[Dict]:
    all_dataset = all_semcor_dataset(cfg)
    dataset = []
    count = n_samples
    if same_sentence_first:
        indices = list(range(len(all_dataset)))
        random.shuffle(indices)
        for index in indices:
            length = min(count, len(all_dataset[index]))  # noqa
            dataset.extend(all_dataset[index][:length])  # noqa
            count -= length
            if count <= 0: break
    else:
        indices = []
        for s_index in range(len(all_dataset)):
            for c_index in range(len(all_dataset[s_index])):
                indices.append((s_index, c_index))
        sample_indices = random.sample(indices, n_samples)
        for s_index, c_index in sample_indices:
            dataset.append(all_dataset[s_index][c_index])
    return dataset


def semcor_inverted_index(cfg: DictConfig, reload: bool = False) -> Dict:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    inv_index_file = _get_semcor_inverted_index_file(cfg)
    if reload or not inv_index_file.is_file():
        dataset = all_semcor_dataset(cfg)
        inv_index = {}
        for i, sent in enumerate(dataset):
            for j, chunk in enumerate(sent):
                synset = chunk["synset"]
                if not synset:
                    continue
                if inv_index.get(synset, None) is None:
                    inv_index[synset] = [(i, j)]
                else:
                    inv_index[synset].append((i, j))
        with open(inv_index_file, mode='w', encoding="utf-8") as fp:
            json.dump(inv_index, fp)
    else:
        with open(inv_index_file, mode='r', encoding="utf-8") as fp:
            inv_index = json.load(fp)
    return inv_index