import json
import pickle
import statistics
from pathlib import Path
from typing import Tuple

from omegaconf import DictConfig

from calf import logger
from exps import merge_config
from .concept_inference import (
    get_concept_root,
    get_embedding_file,
    get_baseline_file
)
from .prompt import prepare_support_prompt, prepare_one_prompt


def get_result_file(n_demo: int, baseline: bool = False) -> Path:
    path = get_concept_root() / "result"
    path.mkdir(parents=True, exist_ok=True)
    if not baseline:
        file = path / f"exact_match_accuracy_{n_demo}.json"
    else:
        file = path / f"exact_match_accuracy_{baseline}.json"
    return file


def exact_match_accuracy(
        embedding_file: Path,
        match_criteria: str = "synonyms",  # word, synonyms
) -> Tuple[int, int]:
    if embedding_file.is_file():
        with open(embedding_file, mode="rb") as fp:
            results = pickle.load(fp)
    else:
        raise FileNotFoundError(f"File {embedding_file} not exist")

    correct = 0
    for result in results:
        if result["task"] in ["d2w", "ld2w"]:
            gen = result["generation"]
            if match_criteria == "word":
                word = result["query_data"]["word"]
                correct = correct + 1 if gen == word else correct
            else:
                synonyms = result["query_data"]["synonyms"]
                correct = correct + 1 if gen in synonyms else correct
        elif result["task"] == "w2w":
            gen = result["generation"]
            word = result["query_data"]["word"]
            correct = correct + 1 if gen == word else correct
        else:
            description = result["query_data"]["description"]
            generation = result["generation"]
            if generation == description:
                correct += 1
    total = len(results)
    return correct, total


def show_generate_results(cfg: DictConfig) -> None:
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
    n_query = cfg.get("query", 0)
    # - runs
    n_runs = cfg.get("runs", 1)

    # - baseline type: shuffle demo
    baseline = cfg.get("baseline", None)

    # - number of result to be shown
    count = cfg.get("results", 9)

    # loop for all models
    for model_name in model_names:
        for n_demo in n_demos:
            for n_run in range(n_runs):
                file = get_embedding_file(cfg, model_name, task, corpus, n_demo, n_query, n_run)
                if baseline:
                    file = get_baseline_file(
                        cfg, model_name, task, corpus, n_demo, n_query, n_run, baseline
                    )
                with open(file, mode="rb") as fp:
                    results = pickle.load(fp)
                    logger.info("-----------------------------------------------")
                    logger.info(f"Embedding file: {file}")

                # show generate results
                for i, result in enumerate(results):
                    if i > count:
                        break
                    support_prompt = prepare_support_prompt(
                        cfg=cfg,
                        support_dataset=result["support_dataset"]
                    )
                    q_data = result["query_data"]
                    prompt, desc_tag = prepare_one_prompt(
                        cfg=cfg,
                        support_prompt=support_prompt,
                        query_data=q_data
                    )
                    if task in ["d2w", "ld2w", "w2w"]:
                        logger.info(f"# {i} {q_data['word']}")
                        logger.info(f"{prompt} {desc_tag} {result['generation']}")
                        logger.info(f"{q_data['synonyms']}\n")
                    else:
                        logger.info(f"# {i} {q_data['word']}")
                        logger.info(f"{prompt} {result['generation']}")
                        logger.info(f"{q_data['description']}\n")


def show_exact_match_results(cfg: DictConfig) -> None:
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
    n_query = cfg.get("query", 0)
    # - runs
    n_runs = cfg.get("runs", 1)

    # - match
    match = cfg.get("match", "synonyms")

    # - baseline type: shuffle demo
    baseline = cfg.get("baseline", None)

    # loop for all models
    results = []
    for model_name in model_names:
        for n_demo in n_demos:
            accuracy = []
            for n_run in range(n_runs):
                file = get_embedding_file(cfg, model_name, task, corpus, n_demo, n_query, n_run)
                if baseline:
                    file = get_baseline_file(
                        cfg, model_name, task, corpus, n_demo, n_query, n_run, baseline
                    )
                correct, total = exact_match_accuracy(file, match)
                accuracy.append(correct/total)
                results.append({
                    "file": str(file),
                    "model": model_name,
                    "demos": n_demo,
                    "baseline": baseline,
                    "n_run": n_run,
                    "correct": correct,
                    "total": total,
                })
                logger.info(f"# {n_run}: {file.name}")
                logger.info(f"exact match accuracy: {correct} / {total} = {correct / total}")
            logger.info(f"Total accuracy: {statistics.mean(accuracy)}")
            results_file = get_result_file(n_demo, baseline)
            with open(results_file, mode='w', encoding="utf-8") as fp:
                json.dump(results, fp)
    del results