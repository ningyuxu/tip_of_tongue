import sys
import random
import math

from omegaconf import DictConfig

from calf import logger


def load_corpus(cfg: DictConfig):
    corpus_name = cfg.get("name", "wordnet")
    getattr(sys.modules[__name__], corpus_name)(cfg)


def concrete(cfg: DictConfig) -> None:
    from . import all_concrete_words, concrete_filter
    func = cfg.get("func", "all")
    if func == "filter":
        dataset = all_concrete_words()
        dataset = concrete_filter(
            dataset,
            bigram=cfg.get("bigram", None),
            min_conc_m=cfg.get("min_conc_m", 0.0),
            max_conc_m=cfg.get("max_conc_m", 5.0),
            percent_known=cfg.get("percent_known", 0.0),
            min_subtlex=cfg.get("min_subtlex", 0),
            max_subtlex=cfg.get("max_subtlex", math.inf),
            pos=cfg.get("pos", None),
            min_word_f=cfg.get("min_word_f", 0.0),
            max_word_f=cfg.get("max_word_f", 1.0),
            min_zipf_f=cfg.get("min_zipf_f", 0.0),
            max_zipf_f=cfg.get("max_zipf_f", 8.0),
        )
    else:  # all
        dataset = all_concrete_words()
    logger.info(f"Load {len(dataset)} words from Concreteness corpus.")
    if dataset:
        logger.info(f"- example: {dataset[0]}")


def semcor(cfg: DictConfig) -> None:
    from . import sample_semcor_dataset, semcor_inverted_index, all_semcor_dataset
    func = cfg.get("func", "all")
    if func == "sample":
        dataset = sample_semcor_dataset(
            cfg=cfg,
            n_samples=cfg.get("n_samples", 2),
            same_sentence_first=cfg.get("same_sentence_first", True)
        )
        n_sentence = len(dataset)
        n_chunks = sum([len(s) for s in dataset])
        logger.info(f"Total {n_sentence} sentences, {n_chunks} chunks")
        logger.info(dataset)
    elif func == "index":
        inv_index_table = semcor_inverted_index(cfg, reload=True)
        synset = cfg.get("synset", "animal.n.01")
        index = random.sample(inv_index_table[synset], k=1)
        dataset = all_semcor_dataset(cfg)
        logger.info(dataset[index[0][0]][index[0][1]])
    else:  # all
        dataset = all_semcor_dataset(cfg, reload=True)
        n_sentence = len(dataset)
        n_chunks = sum([len(s) for s in dataset])
        logger.info(f"Total {n_sentence} sentences, {n_chunks} chunks")


def things(cfg: DictConfig) -> None:
    from . import all_things_dataset
    dataset = all_things_dataset(cfg.get("fmt", "wordnet"))
    logger.info(f"Load {len(dataset)} things from things corpus.")
    logger.info(f"- example: {dataset[0]}")


def revdict(cfg: DictConfig) -> None:
    from . import all_revdic_dataset
    dataset = all_revdic_dataset(cfg)
    logger.info(f"Load {len(dataset)} things from reverse dictionary corpus.")
    logger.info(f"- example: {dataset[0]}")


def wordnet(cfg: DictConfig) -> None:
    from . import (
        wordnet_tree_dataset,
        wordnet_bc5000_dataset,
        wordnet_thing_dataset,
        wordnet_word_dataset,
        wordnet_all_dataset
    )
    func = cfg.get("func", None)
    if func == "tree":
        dataset = wordnet_tree_dataset(
            cfg.get("root_synset", "animal.n.01"),
            cfg.get("tree_depth", 1),
        )
    elif func == "bc5000":
        dataset = wordnet_bc5000_dataset()
    elif func == "things":
        dataset = wordnet_thing_dataset()
    elif func == "word":
        dataset = wordnet_word_dataset(
            cfg.get("word", "bank")
        )
    else:  # func == "all"
        dataset = wordnet_all_dataset(cfg.cache_path, reload=True)
    n_with_examples = 0
    for data in dataset:
        # logger.info(f"{data['synset']}, {data['word']}, {data['lemmas']}")
        if data["examples"]:
            n_with_examples += 1
    logger.info(
        f"total {len(dataset)} samples, {n_with_examples} with examples"
    )


def protoqa(cfg: DictConfig) -> None:
    from . import all_protoqa_dataset
    dataset = all_protoqa_dataset(cfg)
    logger.info(f"Load {len(dataset)} data from protoqa corpus.")
    logger.info(f"- example: {dataset[0]}")


def arce(cfg: DictConfig) -> None:
    from . import all_arc_dataset
    dataset = all_arc_dataset(cfg, "easy")
    logger.info(f"Load {len(dataset)} words from ARC-e corpus.")
    logger.info(f"- example: {dataset[0]}")


def arcc(cfg: DictConfig) -> None:
    from . import all_arc_dataset
    dataset = all_arc_dataset(cfg, "challenge")
    logger.info(f"Load {len(dataset)} words from ARC-c corpus.")
    logger.info(f"- example: {dataset[0]}")


def hellaswag(cfg: DictConfig) -> None:
    from . import all_hellaswag_dataset
    dataset = all_hellaswag_dataset(cfg)
    logger.info(f"Load {len(dataset)} words from HellaSWAG corpus.")
    logger.info(f"- example: {dataset[0]}")


def piqa(cfg: DictConfig) -> None:
    from . import all_piqa_dataset
    dataset = all_piqa_dataset(cfg)
    logger.info(f"Load {len(dataset)} words from PIQA corpus.")
    logger.info(f"- example: {dataset[0]}")


def siqa(cfg: DictConfig) -> None:
    from . import all_siqa_dataset
    dataset = all_siqa_dataset(cfg)
    logger.info(f"Load {len(dataset)} words from SIQA corpus.")
    logger.info(f"- example: {dataset[0]}")


def openbookqa(cfg: DictConfig) -> None:
    from . import all_openbookqa_dataset
    dataset = all_openbookqa_dataset(cfg)
    logger.info(f"Load {len(dataset)} words from OpenbookQA corpus.")
    logger.info(f"- example: {dataset[0]}")


def boolq(cfg: DictConfig) -> None:
    from . import all_boolq_dataset
    dataset = all_boolq_dataset(cfg)
    logger.info(f"Load {len(dataset)} words from BoolQ corpus.")
    logger.info(f"- example: {dataset[0]}")


def csqa(cfg: DictConfig) -> None:
    from . import all_csqa_dataset
    dataset = all_csqa_dataset(cfg)
    logger.info(f"Load {len(dataset)} words from CommonSenseQA corpus.")
    logger.info(f"- example: {dataset[0]}")


def blimp(cfg: DictConfig) -> None:
    from . import all_blimp_dataset
    dataset = all_blimp_dataset(cfg)
    logger.info(f"Load {len(dataset)} words from Blimp corpus.")
    logger.info(f"- example: {dataset['adjunct_island'][0]}")


def sg_eval(cfg: DictConfig) -> None:
    from . import all_sg_eval_dataset
    dataset = all_sg_eval_dataset(cfg)
    logger.info(f"Load {len(dataset)} words from Blimp corpus.")
    logger.info(f"- example: {dataset['reflexive_src_masc'][0]}")


def cslb(cfg: DictConfig) -> None:
    from . import xcslb_dataset_for_feature
    dataset = xcslb_dataset_for_feature()
    logger.info(f"Load {len(dataset)} words from CSLB corpus.")
    logger.info(f"- example: {dataset[100]}")