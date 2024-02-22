import pickle
import random
from pathlib import Path
from typing import List
from omegaconf import DictConfig

from calf import logger, OUTPUT
from calf.utils.log import progress_bar
from calf.modules.huggingface import HuggingfaceModel, HuggingfaceTokenizer
from exps import merge_config
from exps.utils import prompt_inference
from ..utils import get_model_and_tokenizer, trim_generation
from ..corpus.things import all_things_dataset
from .dataset import format_wordnet_dataset
from .prompt import prepare_support_prompt, prepare_one_prompt


# -------------------- Files Functions --------------------
def get_concept_root() -> Path:
    path = OUTPUT / "concept"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_embedding_file_path(model_name: str) -> Path:
    path = get_concept_root() / model_name / "embedding"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_embedding_file(
        cfg: DictConfig,
        model_name: str,
        task: str,
        corpus: str,
        n_demo: int,
        n_query: int,
        n_run: int,
) -> Path:
    model_name = cfg[model_name].name.split('/')[-1]
    path = get_embedding_file_path(model_name)
    file = f"{task}_{corpus}_{n_demo}_{n_query}_{n_run}.pkl"
    return path / file


def get_baseline_file(
        cfg: DictConfig,
        model_name: str,
        task: str,
        corpus: str,
        n_demo: int,
        n_query: int,
        n_run: int,
        baseline: str,
) -> Path:
    model_name = cfg[model_name].name.split('/')[-1]
    path = get_embedding_file_path(model_name)
    file = f"{task}_{corpus}_{n_demo}_{n_query}_{n_run}_{baseline}.pkl"
    return path / file


# -------------------- Dataset Functions --------------------
def prepare_support_dataset(n_demo: int, shuffle: bool = False) -> List:
    all_dataset = format_wordnet_dataset(all_things_dataset(fmt="wordnet"))
    if shuffle:
        all_dataset = shuffle_dataset(all_dataset)
    indices = list(range(len(all_dataset)))
    random.shuffle(indices)
    return [all_dataset[i] for i in indices[:n_demo]]


def prepare_query_dataset(n_query: int, shuffle: bool = False) -> List:
    all_dataset = format_wordnet_dataset(all_things_dataset(fmt="wordnet"))
    if shuffle:
        all_dataset = shuffle_dataset(all_dataset)
    indices = list(range(len(all_dataset)))
    random.shuffle(indices)
    n_query = len(indices) if n_query == 0 else n_query
    return [all_dataset[i] for i in indices[:n_query]]


def shuffle_dataset(dataset: List) -> List:
    length = len(dataset)
    if length > 1:
        random.shuffle(dataset)
        words = [d["word"] for d in dataset]
        offset = random.randint(1, length - 1)
        shuffled_dataset = []
        for i, data in enumerate(dataset):
            shuffled_dataset.append({
                "synset": data["synset"],
                "word": words[(i + offset) % length],
                "description": data["description"],
                "synonyms": data["synonyms"],
                "example": data["example"]
            })
    else:
        shuffled_dataset = dataset
    return shuffled_dataset


def mismatch_dataset(all_dataset: List, support_dataset: List) -> List:
    all_words = [d["word"] for d in all_dataset]
    s_words = [d["word"] for d in support_dataset]
    left_indice = [i for i, w in enumerate(all_words) if w not in s_words]
    left_dataset = [all_dataset[i] for i in left_indice]
    sample_dataset = random.sample(left_dataset, k=len(support_dataset))
    mismatched_dataset = []
    for i, data in enumerate(support_dataset):
        mismatched_dataset.append({
            "synset": data["synset"],
            "word": sample_dataset[i]["word"],
            "description": data["description"],
            "synonyms": data["synonyms"],
            "example": data["example"]
        })
    return mismatched_dataset


# -------------------- Model Functions --------------------
def generate(
        cfg: DictConfig,
        model: HuggingfaceModel,
        tokenizer: HuggingfaceTokenizer,
        support_dataset: List,
        query_dataset: List
) -> List:
    results = []
    support_prompt = prepare_support_prompt(cfg, support_dataset)
    for q_data in progress_bar(logger, query_dataset):
        prompt, desc_tag = prepare_one_prompt(cfg, support_prompt, q_data)
        gen_tokens, prompt_embeddings, gen_embeddings = prompt_inference(
            model=model,
            tokenizer=tokenizer,
            prompt_text=[prompt, f" {desc_tag}"],
            max_new_tokens=cfg.max_new_tokens
        )
        if gen_tokens is None:
            continue

        generation = trim_generation(gen_tokens)
        tokenized_generation = tokenizer(generation.split())
        gener_embeddings = [gen_embeddings[s[0]: s[1]] for s in tokenized_generation["segment_spans"]]
        del gen_embeddings

        prompt_embedding = prompt_embeddings[0][-1]
        arrow_embedding = prompt_embeddings[1][-1]
        del prompt_embeddings

        results.append({
            "task": cfg.embedding.task,
            "support_dataset": support_dataset,
            "query_data": q_data,
            "generation": generation,
            "prompt_embedding": prompt_embedding,
            "arrow_embedding": arrow_embedding,
            "gener_embeddings": gener_embeddings,
        })
    return results


# -------------------- Main Functions --------------------
def run_embed_concept(cfg: DictConfig) -> None:
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
                embedding_file = get_embedding_file(
                    cfg, model_name, task, corpus, n_demo, n_queries, n_run
                )
                logger.info(f"Embedding file: {embedding_file}")
                if not embedding_file.is_file():
                    support_dataset = prepare_support_dataset(n_demo)
                    query_dataset = prepare_query_dataset(n_queries)
                    results = generate(
                        cfg=cfg,
                        model=model,
                        tokenizer=tokenizer,
                        support_dataset=support_dataset,
                        query_dataset=query_dataset
                    )
                    with open(embedding_file, mode="wb") as fp:
                        pickle.dump(results, fp)
                        logger.info(f"Save embedding to {embedding_file.name}")
                    del support_dataset
                    del query_dataset
                    del results
                else:
                    logger.info(f"Embedding file {embedding_file} already exist")
        del model


def run_clone_concept(cfg: DictConfig) -> None:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    # - models
    base_model_name = cfg.get("base_model", "llama2_13b")
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
                base_embedding_file = get_embedding_file(
                    cfg, base_model_name, task, corpus, n_demo, n_queries, n_run
                )
                if base_embedding_file.is_file():
                    embedding_file = get_embedding_file(
                        cfg, model_name, task, corpus, n_demo, n_queries, n_run
                    )
                    logger.info(f"Embedding file: {embedding_file}")
                    if not embedding_file.is_file():
                        with open(base_embedding_file, mode="rb") as fp:
                            logger.info(f"Load from {base_embedding_file.name}")
                            base_results = pickle.load(fp)
                            support_dataset = base_results[0]["support_dataset"]
                            query_dataset = [r["query_data"] for r in base_results]

                        results = generate(
                            cfg=cfg,
                            model=model,
                            tokenizer=tokenizer,
                            support_dataset=support_dataset,
                            query_dataset=query_dataset
                        )
                        with open(embedding_file, mode="wb") as fp:
                            pickle.dump(results, fp)
                            logger.info(f"Save embedding to {embedding_file.name}")
                        del support_dataset
                        del query_dataset
                        del results
                    else:
                        logger.info(f"Embedding file {embedding_file} already exist")
                else:
                    raise FileNotFoundError(f"File {base_embedding_file} not exist")
        del model


def run_concept_baseline(cfg: DictConfig) -> None:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)

    # - baseline type: shuffle demo
    baselines = cfg.get("baseline", "shuffle")
    if isinstance(baselines, str):
        baselines = [baselines]
    for baseline in baselines:
        assert baseline in ["shuffle", "mismatch", "random"]

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
                embedding_file = get_embedding_file(
                    cfg, model_name, task, corpus, n_demo, n_queries, n_run
                )
                if embedding_file.is_file():
                    for baseline in baselines:
                        baseline_file = get_baseline_file(
                            cfg, model_name, task, corpus, n_demo, n_queries, n_run, baseline
                        )
                        logger.info(f"Embedding file: {baseline_file}")
                        if not baseline_file.is_file():
                            with open(embedding_file, mode="rb") as fp:
                                logger.info(f"Load from {embedding_file.name}")
                                base_results = pickle.load(fp)
                                support_dataset = base_results[0]["support_dataset"]
                                query_dataset = [r["query_data"] for r in base_results]

                            if baseline == "shuffle":
                                support_dataset = shuffle_dataset(support_dataset)
                            elif baseline == "mismatch":
                                dataset = format_wordnet_dataset(all_things_dataset(fmt="wordnet"))
                                support_dataset = mismatch_dataset(
                                    all_dataset=dataset,
                                    support_dataset=support_dataset,
                                )
                            elif baseline == "random":
                                support_dataset = prepare_support_dataset(n_demo, shuffle=True)
                                query_dataset = prepare_query_dataset(n_queries, shuffle=True)

                            results = generate(
                                cfg=cfg,
                                model=model,
                                tokenizer=tokenizer,
                                support_dataset=support_dataset,
                                query_dataset=query_dataset
                            )
                            with open(baseline_file, mode="wb") as fp:
                                pickle.dump(results, fp)
                                logger.info(f"Save embedding to {baseline_file.name}")
                            del support_dataset
                            del query_dataset
                            del results
                        else:
                            logger.info(f"Embedding file {baseline_file} already exist")
                else:
                    raise FileNotFoundError(f"File {embedding_file} not exist")
        del model
