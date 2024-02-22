import json
import random
import pickle
from pathlib import Path
from typing import List

from omegaconf import DictConfig
import numpy as np

from calf import logger
from calf.utils.log import progress_bar
from calf.modules.huggingface import (
    HuggingfaceModel,
    HuggingfaceTokenizer,
)
from exps import merge_config
from exps.utils import answer_probability_for_commqa
from exps.concept.dataset import format_wordnet_dataset
from exps.concept.prompt import fill_in_template
from ..corpus import all_things_dataset
from .utils import (
    get_commqa_root,
    get_model_and_tokenizer,
    get_commqa_result_file
)
from .natural_commqa import (
    get_natural_task_probability_file,
    get_support_template as get_natural_s_template,
    build_support_prompt as build_natural_s_prompt,
    get_query_template as get_natural_q_template,
    build_query_prompt as build_natural_q_prompt,
)


# -------------------- File Functions --------------------
def get_concept_task_probability_file(
        model_name: str,
        corpus: str,
        is_simple: bool,
        n_corpus_demos: int,
        n_demos: int,
        n_queries: int,
        run: int,
        task: str,
) -> Path:
    path = get_commqa_root() / corpus
    path.mkdir(parents=True, exist_ok=True)
    file = f"{task}_{model_name}_{is_simple}_{n_corpus_demos}_{n_demos}_{n_queries}_{run}.pkl"
    return path / file


# -------------------- Dataset Functions --------------------
def prepare_support_dataset(n_demo: int) -> List:
    all_dataset = format_wordnet_dataset(all_things_dataset(fmt="wordnet"))
    indices = list(range(len(all_dataset)))
    random.shuffle(indices)
    return [all_dataset[i] for i in indices[:n_demo]]


def mismatch_support_dataset(support_dataset: List) -> List:
    all_dataset = format_wordnet_dataset(all_things_dataset(fmt="wordnet"))
    all_words = [d["word"] for d in all_dataset]
    s_words = [d["word"] for d in support_dataset]
    left_indice = [i for i, w in enumerate(all_words) if w not in s_words]
    left_dataset = [all_dataset[i] for i in left_indice]
    sample_dataset = random.sample(left_dataset, k=len(support_dataset))
    mismatch_dataset = []
    for i, data in enumerate(support_dataset):
        mismatch_dataset.append({
            "synset": data["synset"],
            "word": sample_dataset[i]["word"],
            "description": data["description"],
            "synonyms": data["synonyms"],
            "example": data["example"]
        })
    return mismatch_dataset


def shuffle_support_dataset(support_dataset: List) -> List:
    words = [data["word"] for data in support_dataset]
    shuffled_dataset = []
    for i, data in enumerate(support_dataset):
        shuffled_dataset.append({
            "synset": data["synset"],
            "word": words[(i + 1) % len(words)],
            "description": data["description"],
            "synonyms": data["synonyms"],
            "example": data["example"]
        })
    return shuffled_dataset


def w2w_support_dataset(support_dataset: List) -> List:
    w2w_dataset = []
    for i, data in enumerate(support_dataset):
        w2w_dataset.append({
            "synset": data["synset"],
            "word": data["word"],
            "description": data["word"],
            "synonyms": data["synonyms"],
            "example": data["example"]
        })
    return w2w_dataset


# -------------------- Prompt Functions --------------------
def get_support_template(cfg: DictConfig) -> str:
    desc_tag = cfg.t_desc
    support_template = f"$description {desc_tag} $word"
    return support_template


def prepare_support_prompt(cfg: DictConfig, support_dataset: List) -> str:
    template = get_support_template(cfg)
    prompt = ''
    for data in support_dataset:
        demo = fill_in_template(cfg, data, template)
        prompt = f"{prompt}\n\n{demo}" if prompt else f"{demo}"
    return prompt


# -------------------- Model Functions --------------------
def answer_one_question(
        model: HuggingfaceModel,
        tokenizer: HuggingfaceTokenizer,
        support_prompt: str,
        query_prompt: str,
        desc_tag: str,
        choices: List,
) -> List:
    choice_probs = []
    prompt = f"{support_prompt}\n\n{query_prompt}" if support_prompt else query_prompt
    for choice in choices:
        # choice_words = [f"{w}" for w in choice.split()]
        probability = answer_probability_for_commqa(
            model=model,
            tokenizer=tokenizer,
            prompt_text=f"{prompt} {desc_tag}",
            answer_text=choice,
        )
        choice_probs.append({"choice": choice, "probability": probability})
    return choice_probs


# -------------------- Main Functions --------------------
def calc_commqa_probability(cfg: DictConfig) -> None:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    # - models
    default_model = "llama2_13b"
    model_names = cfg.get("model", default_model)
    if isinstance(model_names, str):
        model_names = [model_names]
    # - corpus
    corpora = cfg.get("corpus", "arce")
    if isinstance(corpora, str):
        corpora = [corpora]
    # - concept demos
    n_demos = cfg.get("demos", 0)
    if isinstance(n_demos, int):
        n_demos = [n_demos]
    # - corpus demos
    n_corpus_demos = cfg.get("corpus_demos", 0)
    # - dataset
    n_queries = cfg.get("queries", 1000)
    # - runs
    n_runs = cfg.get("runs", 1)
    # - task
    task = cfg.get("task", "concept")
    # - use simple template
    is_simple = cfg.get("simple", False)
    if is_simple:
        n_corpus_demos = 0

    for model_name in model_names:
        model, tokenizer = get_model_and_tokenizer(cfg[model_name], cfg.model_dtype)
        for corpus in corpora:
            for n_demo in n_demos:
                for n_run in range(n_runs):
                    hf_model_name = cfg[model_name].name.split('/')[-1]
                    concept_file = get_concept_task_probability_file(
                        model_name=hf_model_name,
                        corpus=corpus,
                        is_simple=is_simple,
                        n_corpus_demos=n_corpus_demos,
                        n_demos=n_demo,
                        n_queries=n_queries,
                        run=n_run,
                        task=task
                    )
                    if not concept_file.is_file():
                        natural_file = get_natural_task_probability_file(
                            model_name=hf_model_name,
                            corpus=corpus,
                            is_simple=False,
                            n_demos=0,
                            n_queries=n_queries,
                            run=n_run,
                            task="natural"
                        )
                        concept_results = []
                        if natural_file.is_file():
                            with open(natural_file, mode="rb") as fp:
                                support_dataset = prepare_support_dataset(n_demo)
                                if task == "shuffle":
                                    support_dataset = shuffle_support_dataset(support_dataset)
                                elif task == "mismatch":
                                    support_dataset = mismatch_support_dataset(support_dataset)
                                elif task == "w2w":
                                    support_dataset = w2w_support_dataset(support_dataset)
                                else:
                                    ...
                                support_prompt = prepare_support_prompt(cfg, support_dataset)
                                # read natural task dataset
                                natural_results = pickle.load(fp)
                                for result in progress_bar(
                                        logger=logger,
                                        iterator=natural_results,
                                        desc=f"# {n_run} {model_name} for {corpus} with {n_demo} demos"
                                ):
                                    natural_s_dataset = result["support_dataset"]
                                    natural_s_template = get_natural_s_template(cfg, corpus, is_simple)
                                    natural_s_prompt = build_natural_s_prompt(
                                        template=natural_s_template,
                                        support_dataset=natural_s_dataset
                                    )

                                    natural_q_data = result["query_data"]
                                    natural_q_template = get_natural_q_template(cfg, corpus, is_simple)
                                    natural_q_prompt = build_natural_q_prompt(
                                        template=natural_q_template,
                                        support_prompt=natural_s_prompt,
                                        query_data=natural_q_data
                                    )

                                    desc_tag = cfg.t_desc
                                    choice_probs = answer_one_question(
                                        model=model,
                                        tokenizer=tokenizer,
                                        support_prompt=support_prompt,
                                        query_prompt=natural_q_prompt,
                                        desc_tag=desc_tag,
                                        choices=natural_q_data["choices"]
                                    )
                                    concept_results.append({
                                        "corpus": corpus,
                                        "support_dataset": support_dataset,
                                        "query_data": natural_q_data,
                                        "choice_probs": choice_probs,
                                    })
                                del natural_results
                            with open(concept_file, mode="wb") as fp:
                                pickle.dump(concept_results, fp)
                                logger.info(f"Save results to {concept_file.name}")
                            del concept_results
                        else:
                            raise FileNotFoundError(f"File {natural_file} not exist")
                    else:
                        logger.info(f"File {concept_file.name} already exist")
        del model


def calc_commqa_accuracy(cfg: DictConfig) -> None:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    # - models
    default_model = "llama2_13b"
    model_names = cfg.get("model", default_model)
    if isinstance(model_names, str):
        model_names = [model_names]
    # - corpus
    corpora = cfg.get("corpus", "arce")
    if isinstance(corpora, str):
        corpora = [corpora]
    # - demos
    n_demos = cfg.get("demos", 0)
    # - corpus demos
    n_corpus_demos = cfg.get("corpus_demos", 0)
    if isinstance(n_demos, int):
        n_demos = [n_demos]
    # - dataset
    n_queries = cfg.get("queries", 1000)
    # - runs
    n_runs = cfg.get("runs", 1)
    # - task
    task = cfg.get("task", "concept")
    # - use simple template
    is_simple = cfg.get("simple", False)
    if is_simple:
        n_corpus_demos = 0

    prob_results = []
    for model_name in model_names:
        for corpus in corpora:
            for n_demo in n_demos:
                correct, total = 0, 0
                for n_run in range(n_runs):
                    hf_model_name = cfg[model_name].name.split('/')[-1]
                    concept_file = get_concept_task_probability_file(
                        model_name=hf_model_name,
                        corpus=corpus,
                        is_simple=is_simple,
                        n_corpus_demos=n_corpus_demos,
                        n_demos=n_demo,
                        n_queries=n_queries,
                        run=n_run,
                        task=task
                    )
                    if concept_file.is_file():
                        with open(concept_file, mode="rb") as fp:
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
                    else:
                        raise FileNotFoundError(f"File {concept_file} not found")
                prob_results.append({
                    "model_name": model_name,
                    "corpus": corpus,
                    "n_demo": n_demo,
                    "n_queries": n_queries,
                    "correct": correct,
                    "total": total
                })
                logger.info(f"# {model_name} | {corpus} | {n_demo}")
                logger.info(f"- accuracy: {correct} / {total} = {correct / total}")
    result_file = get_commqa_result_file(task)
    with open(result_file, mode='w', encoding="utf-8") as fp:
        json.dump(prob_results, fp, indent=2)
