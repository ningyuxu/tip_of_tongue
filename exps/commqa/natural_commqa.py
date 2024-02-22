import json
import pickle
import random
from pathlib import Path
from typing import List, Dict

from omegaconf import DictConfig
import numpy as np

from calf import logger
from calf.utils.log import progress_bar
from calf.modules import HuggingfaceModel, HuggingfaceTokenizer
from exps import merge_config
from exps.utils import answer_probability_for_commqa
from ..corpus import (
    all_arc_dataset,
    all_hellaswag_dataset,
    all_piqa_dataset,
    all_siqa_dataset,
    all_openbookqa_dataset,
    all_boolq_dataset,
    all_csqa_dataset,
)
from .utils import get_commqa_root, get_model_and_tokenizer, get_commqa_result_file


# -------------------- File Functions --------------------
def get_query_dataset_file(corpus: str, n_queries: int, n_run: int) -> Path:
    path = get_commqa_root() / "dataset"
    path.mkdir(parents=True, exist_ok=True)
    file = f"query_dataset_{corpus}_{n_queries}_{n_run}.pkl"
    return path / file


def get_natural_task_probability_file(
        model_name: str,
        corpus: str,
        is_simple: bool,
        n_demos: int,
        n_queries: int,
        run: int,
        task: str,
) -> Path:
    path = get_commqa_root() / corpus
    path.mkdir(parents=True, exist_ok=True)
    file = f"{task}_{model_name}_{is_simple}_{n_demos}_{n_queries}_{run}.pkl"
    return path / file


# -------------------- Dataset Functions --------------------
def prepare_dataset(cfg: DictConfig, corpus: str) -> List:
    if corpus == "arce":
        dataset = all_arc_dataset(cfg, "easy")
    elif corpus == "arcc":
        dataset = all_arc_dataset(cfg, "challenge")
    elif corpus == "hellaswag":
        dataset = all_hellaswag_dataset(cfg)
    elif corpus == "piqa":
        dataset = all_piqa_dataset(cfg)
    elif corpus == "siqa":
        dataset = all_siqa_dataset(cfg)
    elif corpus == "openbookqa":
        dataset = all_openbookqa_dataset(cfg)
    elif corpus == "boolq":
        dataset = all_boolq_dataset(cfg)
    elif corpus == "csqa":
        dataset = all_csqa_dataset(cfg)
    else:
        raise ValueError(f"Unrecognized corpus {corpus}")
    return dataset


def prepare_query_dataset(cfg: DictConfig, corpus: str, n_queries: int, n_run: int) -> List:
    query_datasete_file = get_query_dataset_file(corpus, n_queries, n_run)
    if not query_datasete_file.is_file():
        all_dataset = prepare_dataset(cfg, corpus)
        indices = list(range(len(all_dataset)))
        random.shuffle(indices)
        if n_queries == 0:
            n_queries = len(indices)
        dataset = [all_dataset[i] for i in indices[:n_queries]]
        with open(query_datasete_file, mode='w', encoding="utf-8") as fp:
            json.dump(dataset, fp)
    else:
        with open(query_datasete_file, mode='r', encoding="utf-8") as fp:
            dataset = json.load(fp)
    return dataset


def prepare_support_dataset(cfg: DictConfig, n_demos: int, corpus: str) -> List:
    all_dataset = prepare_dataset(cfg, corpus)
    indices = list(range(len(all_dataset)))
    random.shuffle(indices)
    return [all_dataset[i] for i in indices[:n_demos]]


# -------------------- Prompt Functions --------------------
def get_support_template(cfg: DictConfig, corpus: str, simple_template: bool = False) -> str:
    if corpus in ["arce", "arcc"]:
        template = cfg.arc_s_template if not simple_template else cfg.arc_s_template_simple
    elif corpus == "hellaswag":
        template = cfg.hellaswag_s_template if not simple_template else cfg.hellaswag_s_template_simple
    elif corpus == "piqa":
        template = cfg.piqa_s_template if not simple_template else cfg.piqa_s_template_simple
    elif corpus == "siqa":
        template = cfg.siqa_s_template if not simple_template else cfg.siqa_s_template_simple
    elif corpus == "openbookqa":
        template = cfg.obqa_s_template if not simple_template else cfg.obqa_s_template_simple
    elif corpus == "boolq":
        template = cfg.boolq_s_template if not simple_template else cfg.boolq_s_template_simple
    elif corpus == "csqa":
        template = cfg.csqa_s_template if not simple_template else cfg.csqa_s_template_simple
    else:
        raise ValueError(f"Corpus {corpus} not supported")
    return template


def get_query_template(cfg: DictConfig, corpus: str = None, simple_template: bool = False) -> str:
    if corpus in ["arce", "arcc"]:
        template = cfg.arc_q_template if not simple_template else cfg.arc_q_template_simple
    elif corpus == "hellaswag":
        template = cfg.hellaswag_q_template if not simple_template else cfg.hellaswag_q_template_simple
    elif corpus == "piqa":
        template = cfg.piqa_q_template if not simple_template else cfg.piqa_q_template_simple
    elif corpus == "siqa":
        template = cfg.siqa_q_template if not simple_template else cfg.siqa_q_template_simple
    elif corpus == "openbookqa":
        template = cfg.obqa_q_template if not simple_template else cfg.obqa_q_template_simple
    elif corpus == "boolq":
        template = cfg.boolq_q_template if not simple_template else cfg.boolq_q_template_simple
    elif corpus == "csqa":
        template = cfg.csqa_q_template if not simple_template else cfg.csqa_q_template_simple
    else:
        raise ValueError(f"Corpus {corpus} not supported")
    return template


def fill_in_template(data: Dict, template: str) -> str:
    context = data["context"]
    question = data["question"]
    choices = data["choices"]
    answer_index = data["answer_index"]
    answer = choices[answer_index]
    result = template.replace("$context", context)
    result = result.replace("$question", question)
    result = result.replace("$answer", answer)
    return result


def build_support_prompt(template: str, support_dataset: List) -> str:
    prompt = ''
    for data in support_dataset:
        demo = fill_in_template(data, template)
        prompt = f"{prompt}\n{demo}" if prompt else f"{demo}"
    return prompt


def build_query_prompt(template: str, support_prompt: str, query_data: Dict) -> str:
    q_prompt = fill_in_template(query_data, template)
    prompt = f"{support_prompt}\n{q_prompt}" if support_prompt else q_prompt
    return prompt


# -------------------- Model Functions --------------------
def answer_one_question(
        model: HuggingfaceModel,
        tokenizer: HuggingfaceTokenizer,
        prompt: str,
        choices: List,
) -> List:
    choice_probs = []
    for choice in choices:
        # choice_words = [f"{w}" for w in choice.split()]
        probability = answer_probability_for_commqa(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            answer_text=choice,
        )
        choice_probs.append({"choice": choice, "probability": probability})
    return choice_probs


# -------------------- Main Functions --------------------
def calc_commqa_probability(cfg: DictConfig) -> None:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    # - models
    model_names = cfg.get("model", "llama2_13b")
    if isinstance(model_names, str):
        model_names = [model_names]
    # - corpus
    corpora = cfg.get("corpus", "arce")
    if isinstance(corpora, str):
        corpora = [corpora]
    # - demos
    n_demos = cfg.get("demos", 0)
    if isinstance(n_demos, int):
        n_demos = [n_demos]
    # - dataset
    n_queries = cfg.get("queries", 1000)
    # - runs
    n_runs = cfg.get("runs", 1)
    # - task
    task = "natural"
    # - use simple template
    is_simple = cfg.get("simple", False)
    if is_simple:
        n_demos = [0]

    for model_name in model_names:
        model, tokenizer = get_model_and_tokenizer(cfg[model_name], cfg.model_dtype)
        for corpus in corpora:
            for n_demo in n_demos:
                for n_run in range(n_runs):
                    query_dataset = prepare_query_dataset(cfg, corpus, n_queries, n_run)
                    hf_model_name = cfg[model_name].name.split('/')[-1]
                    prob_file = get_natural_task_probability_file(
                        model_name=hf_model_name,
                        corpus=corpus,
                        is_simple=is_simple,
                        n_demos=n_demo,
                        n_queries=n_queries,
                        run=n_run,
                        task=task
                    )
                    if not prob_file.is_file():
                        support_dataset = prepare_support_dataset(cfg, n_demo, corpus)
                        support_template = get_support_template(cfg, corpus, is_simple)
                        support_prompt = build_support_prompt(support_template, support_dataset)
                        results = []
                        for query_data in progress_bar(
                                logger=logger,
                                iterator=query_dataset,
                                desc=f"# {n_run} {model_name} for {corpus} with {n_demo} demos"
                        ):
                            query_template = get_query_template(cfg, corpus, is_simple)
                            prompt = build_query_prompt(query_template, support_prompt, query_data)
                            choice_probs = answer_one_question(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=prompt,
                                choices=query_data["choices"]
                            )
                            results.append({
                                "corpus": corpus,
                                "support_dataset": support_dataset,
                                "query_data": query_data,
                                "choice_probs": choice_probs,
                            })
                        with open(prob_file, mode="wb") as fp:
                            pickle.dump(results, fp)
                            logger.info(
                                f"Save {model_name} for {corpus} with {n_demo} demos "
                                f"results to {prob_file.name}"
                            )
                        del results
                    else:
                        logger.info(
                            f"File {prob_file.name} for {model_name} {corpus} with {n_demo} "
                            f"demos already exist"
                        )
        del model


def calc_commqa_accuracy(cfg: DictConfig) -> None:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    # - models
    model_names = cfg.get("model", "llama2_13b")
    if isinstance(model_names, str):
        model_names = [model_names]
    # - corpus
    corpora = cfg.get("corpus", "arce")
    if isinstance(corpora, str):
        corpora = [corpora]
    # - demos
    n_demos = cfg.get("demos", 0)
    if isinstance(n_demos, int):
        n_demos = [n_demos]
    # - dataset
    n_queries = cfg.get("queries", 1000)
    # - runs
    n_runs = cfg.get("runs", 1)
    # - task
    task = cfg.get("task", "natural")
    # - use simple template
    is_simple = cfg.get("simple", False)
    if is_simple:
        n_demos = [0]

    prob_results = []
    for model_name in model_names:
        for corpus in corpora:
            for n_demo in n_demos:
                for n_run in range(n_runs):
                    correct, total = 0, 0
                    hf_model_name = cfg[model_name].name.split('/')[-1]
                    prob_file = get_natural_task_probability_file(
                        model_name=hf_model_name,
                        corpus=corpus,
                        is_simple=is_simple,
                        n_demos=n_demo,
                        n_queries=n_queries,
                        run=n_run,
                        task=task
                    )
                    if prob_file.is_file():
                        with open(prob_file, mode="rb") as fp:
                            results = pickle.load(fp)
                            for result in results:
                                choice_probs = result["choice_probs"]
                                probability = [
                                    np.sum([np.sum(w_p) for w_p in d["probability"]])
                                    for d in choice_probs
                                ]
                                answer_index = result["query_data"]["answer_index"]
                                predict_index = probability.index(max(probability))
                                if answer_index == predict_index:
                                    correct += 1
                                total += 1
                        prob_results.append({
                            "model_name": model_name,
                            "corpus": corpus,
                            "n_demo": n_demo,
                            "n_run": n_run,
                            "n_queries": n_queries,
                            "correct": correct,
                            "total": total
                        })
                        logger.info(f"# {model_name} | {corpus} | {n_demo}")
                        logger.info(f"- accuracy: {correct} / {total} = {correct / total}")
                    else:
                        raise FileNotFoundError(f"File {prob_file} not found")

    result_file = get_commqa_result_file(task)
    with open(result_file, mode='w', encoding="utf-8") as fp:
        json.dump(prob_results, fp, indent=2)
