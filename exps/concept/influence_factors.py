import json
import pickle
import random
import statistics
from pathlib import Path
from typing import List

from omegaconf import DictConfig
from wordfreq import zipf_frequency
from nltk.corpus import wordnet as wn

from calf import OUTPUT, logger
from calf.utils.common import partial_shuffle
from exps import merge_config
from ..corpus import all_things_dataset, all_revdic_dataset, wordnet_thing_dataset, wordnet_all_dataset
from ..utils import get_model_and_tokenizer
from .dataset import format_wordnet_dataset
from .concept_inference import generate
from .output_results import exact_match_accuracy


# -------------------- Files Functions --------------------
def get_influence_root() -> Path:
    path = OUTPUT / "influence"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_influence_file_path(model_name: str) -> Path:
    path = get_influence_root() / model_name / "embedding"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_desc_influence_file(
        cfg: DictConfig,
        model_name: str,
        task: str,
        corpus: str,
        n_demo: int,
        n_query: int,
        n_run: int,
        desc: str,
        degree: float,
) -> Path:
    model_name = cfg[model_name].name.split('/')[-1]
    path = get_influence_file_path(model_name)
    file_stem = f"{task}_{corpus}_{n_demo}_{n_query}_{n_run}"
    if desc != "shuffle":
        file = f"{file_stem}_desc_{desc}.pkl"
    else:
        file = f"{file_stem}_desc_{desc}_{str(degree)}.pkl"
    return path / file


def get_concept_influ_file(
        cfg: DictConfig,
        model_name: str,
        task: str,
        corpus: str,
        n_demo: int,
        n_query: int,
        n_run: int,
) -> Path:
    model_name = cfg[model_name].name.split('/')[-1]
    path = OUTPUT / "wordnet" / model_name / "embedding"
    file_stem = f"{task}_{corpus}_{n_demo}_{n_query}_{n_run}"
    file = f"{file_stem}_wordnet.pkl"
    return path / file


def prepare_query_dataset(cfg: DictConfig, n_queries: int) -> List:
    dataset = wordnet_all_dataset(cfg.cache_path)
    dataset = format_wordnet_dataset(dataset)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    n_queries = len(indices) if n_queries == 0 else n_queries
    query_dataset = [dataset[i] for i in indices[:n_queries]]
    return query_dataset


def search_influence_concept_files(
        cfg: DictConfig,
        model_name: str,
        task: str,
        corpus: str,
        n_demo: int,
        n_query: int,
        n_run: int,
        file_id: int = None
) -> List[Path]:
    file = get_concept_influ_file(cfg, model_name, task, corpus, n_demo, n_query, n_run)
    path = file.parent
    file_stem = file.stem
    if file_id is None:
        pattern = f"{file_stem}_*.pkl"
    else:
        pattern = f"{file_stem}_{file_id}.pkl"
    files = [f for f in path.glob(pattern) if f.is_file()]
    concept_files = sorted(files, key=lambda f: f.name)
    return concept_files


def get_desc_influ_result_file(match: str) -> Path:
    path = get_influence_root() / "result"
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"desc_influ_match_{match}.json"
    return file


def get_concept_influ_result_file(match: str) -> Path:
    path = get_influence_root() / "result"
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"concept_influ_match_{match}.json"
    return file


# -------------------- Dataset Functions --------------------
def prepare_desc_support_dataset(n_demo: int) -> List:
    all_dataset = format_wordnet_dataset(all_things_dataset(fmt="wordnet"))
    all_dataset = [d for d in all_dataset if d["synset"]]
    indices = list(range(len(all_dataset)))
    random.shuffle(indices)
    return [all_dataset[i] for i in indices[:n_demo]]


def prepare_desc_query_dataset(n_queries: int, desc: str, degree: float) -> List:
    if desc in ["things", "shuffle"]:
        dataset = all_things_dataset(fmt="wordnet")
        dataset = [d for d in dataset if d["synset"]]
    elif desc == "wordnet":
        dataset = wordnet_thing_dataset()
    elif desc == "hill200":
        dataset = all_revdic_dataset()
    else:
        raise ValueError(f"Argument func {desc} is invalid")
    dataset = format_wordnet_dataset(dataset)
    # prepare queries
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    n_queries = len(indices) if n_queries == 0 else n_queries
    query_dataset = [dataset[i] for i in indices[:n_queries]]
    if desc == "shuffle":
        for data in query_dataset:
            description = data["description"]
            desc_words = description.split()
            desc_words = partial_shuffle(desc_words, degree)
            data["description"] = ' '.join(desc_words)
    return query_dataset


def run_influ_description(cfg: DictConfig) -> None:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    # - models
    model_names = cfg.get("model", "llama2_13b")
    if isinstance(model_names, str):
        model_names = [model_names]
    # - task
    task = cfg.get("task", "d2w")
    # - corpus
    corpus = cfg.get("corpus", "things")
    # - demos
    n_demos = cfg.get("demos", 0)
    if isinstance(n_demos, int):
        n_demos = [n_demos]
    # - query
    n_queries = cfg.get("queries", 0)
    # - runs
    n_runs = cfg.get("runs", 1)

    descs = cfg.get("descs", "things")
    if isinstance(descs, str):
        descs = [descs]

    for model_name in model_names:
        model, tokenizer = get_model_and_tokenizer(cfg[model_name], cfg.model_dtype, bos=False)
        for n_demo in n_demos:
            for n_run in range(n_runs):
                support_dataset = prepare_desc_support_dataset(n_demo)
                for desc in descs:
                    degrees = [0.3, 0.6, 1] if desc == "shuffle" else [0]
                    for degree in degrees:
                        influence_file = get_desc_influence_file(
                            cfg, model_name, task, corpus, n_demo, n_queries, n_run, desc, degree
                        )
                        if not influence_file.is_file():
                            query_dataset = prepare_desc_query_dataset(n_queries, desc, degree)
                            results = generate(
                                cfg=cfg,
                                model=model,
                                tokenizer=tokenizer,
                                support_dataset=support_dataset,
                                query_dataset=query_dataset
                            )
                            with open(influence_file, mode="wb") as fp:
                                pickle.dump(results, fp)
                                logger.info(f"Save embedding to {influence_file}")
                            del query_dataset
                            del results
                        else:
                            logger.info(f"Embedding file {influence_file} already exist")
                del support_dataset
        del model


def show_desc_influ_results(cfg: DictConfig) -> None:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    # - models
    model_names = cfg.get("model", "llama2_13b")
    if isinstance(model_names, str):
        model_names = [model_names]
    # - task
    task = cfg.get("task", "d2w")
    # - corpus
    corpus = cfg.get("corpus", "things")
    # - demos
    n_demos = cfg.get("demos", 0)
    if isinstance(n_demos, int):
        n_demos = [n_demos]
    # - query
    n_queries = cfg.get("queries", 0)
    # - runs
    n_runs = cfg.get("runs", 1)

    descs = cfg.get("descs", "things")
    if isinstance(descs, str):
        descs = [descs]

    match = cfg.get("match", "synonyms")

    results_file = get_desc_influ_result_file(match)
    results = []
    for model_name in model_names:
        for n_demo in n_demos:
            for desc in descs:
                degrees = [0.3, 0.6, 1] if desc == "shuffle" else [0]
                for degree in degrees:
                    accuracy = []
                    for n_run in range(n_runs):
                        influence_file = get_desc_influence_file(
                            cfg, model_name, task, corpus, n_demo, n_queries, n_run, desc, degree
                        )
                        correct, total = exact_match_accuracy(influence_file, match)
                        accuracy.append(correct / total)
                        results.append({
                            "file": str(influence_file),
                            "model": model_name,
                            "dataset": desc,
                            "demos": n_demo,
                            "shuffle_degree": degree,
                            "run": n_run,
                            "correct": correct,
                            "total": total,
                        })
                        logger.info(f"# {n_run}: {influence_file}")
                        logger.info(f"exact match accuracy: {correct} / {total} = {correct / total}")
                    logger.info(f"Total accuracy: {statistics.mean(accuracy)}")
    with open(results_file, mode='w', encoding="utf-8") as fp:
        json.dump(results, fp)
    del results


def run_influ_concept(cfg: DictConfig) -> None:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    # - models
    model_names = cfg.get("model", "llama2_13b")
    if isinstance(model_names, str):
        model_names = [model_names]
    # - task
    task = cfg.get("task", "d2w")
    # - corpus
    corpus = cfg.get("corpus", "things")
    # - demos
    n_demos = cfg.get("demos", 0)
    if isinstance(n_demos, int):
        n_demos = [n_demos]
    # - query
    n_queries = cfg.get("queries", 0)
    # - runs
    n_runs = cfg.get("runs", 1)

    for model_name in model_names:
        model, tokenizer = get_model_and_tokenizer(cfg[model_name], cfg.model_dtype, bos=False)
        for n_demo in n_demos:
            for n_run in range(n_runs):
                concept_file = get_concept_influ_file(
                    cfg, model_name, task, corpus, n_demo, n_queries, n_run
                )
                support_dataset = prepare_desc_support_dataset(n_demo)
                query_dataset = prepare_query_dataset(cfg, n_queries)
                for fid in range(len(query_dataset) // 1000 + 1):
                    seq_file = concept_file.parent / f"{concept_file.stem}_{fid}{concept_file.suffix}"
                    if seq_file.is_file():
                        logger.info(f"Embedding file {seq_file} already exist")
                        continue
                    logger.info(f"Concept file: {seq_file}")
                    # split query dataset
                    seq_query_dataset = query_dataset[fid*1000:(fid+1)*1000]
                    results = generate(
                        cfg=cfg,
                        model=model,
                        tokenizer=tokenizer,
                        support_dataset=support_dataset,
                        query_dataset=seq_query_dataset,
                    )
                    with open(seq_file, mode="wb") as fp:
                        pickle.dump(results, fp)
                        logger.info(f"Save embedding to {seq_file.name}")
                del support_dataset
                del query_dataset
                del results
        del model


def get_influence_factors_info(embedding_files: List[Path], match_criteria: str = "synonyms") -> List:
    with open(embedding_files[0], mode="rb") as fp:
        results = pickle.load(fp)
        task = results[0]["task"]
        if task not in ["d2w", "ld2w"]:
            raise ValueError(f"Task {task} not support")

    words_info = []
    for file in embedding_files:
        with open(file, mode="rb") as fp:
            results = pickle.load(fp)
            for result in results:
                synset = result["query_data"]["synset"]
                word = result["query_data"]["word"]
                desc = result["query_data"]["description"]
                synonyms = result["query_data"]["synonyms"]
                gen = result["generation"]
                if match_criteria == "word":
                    is_correct = True if gen == word else False
                else:
                    is_correct = True if gen in synonyms else False
                words_info.append({
                    "synset": synset,
                    "word": word,
                    "correct": is_correct,
                    "frequency": zipf_frequency(word, "en"),
                    "desc_length": len(desc.split()),
                    "sense_number": len(wn.synsets(word.replace(' ', '_')))
                })
    return words_info


def show_concept_influ_results(cfg: DictConfig) -> None:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    # - models
    model_names = cfg.get("model", "llama2_13b")
    if isinstance(model_names, str):
        model_names = [model_names]
    # - task
    task = cfg.get("task", "d2w")
    # - corpus
    corpus = cfg.get("corpus", "things")
    # - demos
    n_demos = cfg.get("demos", 0)
    if isinstance(n_demos, int):
        n_demos = [n_demos]
    # - query
    n_queries = cfg.get("queries", 0)
    # - runs
    n_runs = cfg.get("runs", 1)

    match = cfg.get("match", "synonyms")

    for model_name in model_names:
        model, tokenizer = get_model_and_tokenizer(cfg[model_name], cfg.model_dtype, bos=False)
        for n_demo in n_demos:
            for n_run in range(n_runs):
                # search embedding files
                concept_files = search_influence_concept_files(
                    cfg, model_name, task, corpus, n_demo, n_queries, n_run
                )
                words_info = get_influence_factors_info(concept_files, match)
                correct, total = 0, 0
                for info in words_info:
                    correct = correct + 1 if info["correct"] else correct
                    total += 1
                logger.info(f"{correct} / {total} = {correct / total}")
                results_file = get_concept_influ_result_file(match)
                with open(results_file, mode='w', encoding="utf-8") as fp:
                    json.dump(words_info, fp)
