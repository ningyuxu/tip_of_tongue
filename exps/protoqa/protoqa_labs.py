import re
import json
import random
import collections
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

from omegaconf import DictConfig

from calf import logger, OUTPUT
from calf.utils.log import progress_bar
from calf.modules.huggingface import get_model, get_tokenizer
from exps import merge_config
from exps.corpus.protoqa import all_protoqa_dataset
from exps.utils import sample_sequence
from exps.concept.dataset import format_wordnet_dataset

from exps.corpus import all_things_dataset


# -------------------- File Functions --------------------
def get_protoqa_root() -> Path:
    path = OUTPUT / "protoqa"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_ranked_list_file(
        task: str,
        model_name: str,
        n_demos: int,
        n_run: int,
        greedy: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        split: str = "dev",
) -> Path:
    path = get_protoqa_root() / model_name
    path.mkdir(parents=True, exist_ok=True)
    if greedy:
        file_stem = f"{task}_{n_demos}_{greedy}_{split}_{n_run}"
    else:
        file_stem = f"{task}_{n_demos}_{temperature}_{top_k}_{top_p}_{repetition_penalty}_{split}_{n_run}"
    file = path / f"{file_stem}_ranked_list.jsonl"
    return file


def get_sample_answers_file(
        task: str,
        model_name: str,
        n_demos: int,
        n_run: int,
        greedy: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        split: str = "dev"
) -> Path:
    path = get_protoqa_root() / model_name
    path.mkdir(parents=True, exist_ok=True)
    if greedy:
        file_stem = f"{task}_{n_demos}_{greedy}_{split}_{n_run}"
    else:
        file_stem = f"{task}_{n_demos}_{temperature}_{top_k}_{top_p}_{repetition_penalty}_{split}_{n_run}"
    file = path / f"{file_stem}_sample_answers.jsonl"
    return file


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


def fill_in_template(data: Dict, template: str) -> str:
    word = data["word"]
    description = data["description"]
    result = template.replace("$word", word)
    result = result.replace("$description", description)
    return result


def prepare_support_prompt(cfg: DictConfig, support_dataset: List) -> str:
    template = get_support_template(cfg)
    prompt = ''
    for data in support_dataset:
        demo = fill_in_template(data, template)
        prompt = f"{prompt}\n{demo}" if prompt else f"{demo}"
    return prompt


def format_question(question: str, use_gpt_fmt: bool = False) -> str:
    question = question.lower()
    question = question.replace(',', '')
    question = question.replace('.', '')
    question = question.replace(':', '')
    question = question.replace('?', '')
    if use_gpt_fmt:
        question = question.replace("someone", "one person")
        question = question.replace("someplace", "one place")
        if "name something" in question:
            question = question.replace("name something", "one thing")
            question += " is"
        elif "tell me something" in question:
            question = question.replace("tell me something", "one thing")
            question += " is"
        elif "name a " in question:
            question = question.replace("name a ", "one ")
            question += " is"
        elif "name an " in question:
            question = question.replace("name an ", "one ")
            question += " is"
        elif "name" in question:
            question = question.replace("name", '')
            question += " is"
        elif question.startswith("tell me a "):
            question = question.replace("tell me a ", "one ")
            question += " is"
        elif question.startswith("tell me an "):
            question = question.replace("tell me an ", "one ")
            question += " is"
        elif question.startswith("what "):
            question = question.replace("what", "one")
            question += " is"
        elif question.startswith("give me a "):
            question = question.replace("give me a ", "one ")
            question += " is"
        elif question.startswith("tell me "):
            question = question.replace("tell me ", '')
            question += " is"
        elif "which" in question:
            question = question.replace("which", "one")
            question += " is"
        elif "what" in question:
            question = question.replace("what", "one")
            question += " is"
        elif "how can you tell" in question:
            question = question.replace("how can you tell", "one way to tell")
            question += " is"
        else:
            question = "Q: " + question + "? A: "
    return question


def prepare_query_dataset(cfg: DictConfig, task: str, n_queries: int) -> List:
    all_dataset = all_protoqa_dataset(cfg)
    n_queries = len(all_dataset) if n_queries == 0 else n_queries
    indices = list(range(len(all_dataset)))
    random.shuffle(indices)
    dataset = [all_dataset[i] for i in indices[:n_queries]]
    gpt_fmt = True if task == "natural" else False
    query_dataset = []
    for data in dataset:
        query_dataset.append({
            "data_id": data["data_id"],
            "question": format_question(data["question"], gpt_fmt)
        })
    return query_dataset


# -------------------- Model Functions --------------------
def get_model_and_tokenizer(model_cfg: DictConfig, model_dtype: str) -> Tuple:
    revision = model_cfg.get("revision", None)
    trust_remote_code = model_cfg.get("trust_remote_code", False)
    model = get_model(
        model_name=model_cfg.name,
        revision=revision,
        model_dtype=model_dtype,
        n_context=model_cfg.n_context,
        trust_remote_code=trust_remote_code
    )
    tokenizer = get_tokenizer(
        model_name=model_cfg.name,
        revision=revision,
        trust_remote_code=trust_remote_code
    )
    return model, tokenizer


def trim_generation(generation: str) -> str:
    generation = generation.strip()
    pattern = rf"^(a |an |the )?(.*)$(\r\n?|\n)*?"
    match = re.match(pattern, generation, re.M)
    if match is not None:
        generation = match.group(2)
        generation = re.sub(r"[^\w\s]", '', generation)
        return generation
    else:
        return "<|Invalid|>"


# -------------------- Main Functions --------------------
def sample_protoqa_answers(cfg: DictConfig) -> None:
    cfg = merge_config(str(Path(__file__).parent), "../meaning/concept/config.yaml", cfg)

    model_names = cfg.get("model", "llama2_13b")
    if isinstance(model_names, str):
        model_names = [model_names]
    n_demos = cfg.get("demos", 0)
    if isinstance(n_demos, int):
        n_demos = [n_demos]
    n_queries = cfg.get("queries", 1000)
    task = cfg.get("task", "natural")
    n_samples = cfg.get("samples", 300)
    greedy = cfg.get("greedy", False)
    temperature = cfg.get("temperature", 1)
    top_k = cfg.get("top_k", 0)
    top_p = cfg.get("top_p", 0.0)
    repetition_penalty = cfg.get("penalty", 1.0)
    split = cfg.get("split", "dev")
    n_runs = cfg.get("runs", 1)

    for model_name in model_names:
        model, tokenizer = get_model_and_tokenizer(cfg[model_name], cfg.model_dtype)
        for n_demo in n_demos:
            for n_run in range(n_runs):
                hf_model_name = cfg[model_name].name.split('/')[-1]
                ranked_list_file = get_ranked_list_file(
                    task=task,
                    model_name=hf_model_name,
                    n_demos=n_demo,
                    n_run=n_run,
                    greedy=greedy,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    split=split,
                )
                sample_answers_file = get_sample_answers_file(
                    task=task,
                    model_name=hf_model_name,
                    n_demos=n_demo,
                    n_run=n_run,
                    greedy=greedy,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    split=split,
                )
                if not ranked_list_file.is_file():
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
                    query_dataset = prepare_query_dataset(cfg, task, n_queries)

                    qids = [q["data_id"] for q in query_dataset]
                    questions = [q["question"] for q in query_dataset]
                    prediced_results = collections.defaultdict(list)
                    result = []
                    for seq_id in range(len(questions)):
                        question = questions[seq_id]
                        desc_tag = cfg.t_desc
                        if support_prompt:
                            prompt = f"{support_prompt}\n{question} {desc_tag}"
                        else:
                            prompt = f"{question}"
                        gen_answer_list = []
                        desc = (f"# Q: {seq_id}/{len(questions)} | M: {model_name} "
                                f"| D: {n_demo} | R: {n_run}")
                        for _ in progress_bar(logger, range(n_samples), desc=desc):
                            gen_tokens = sample_sequence(
                                model=model,
                                tokenizer=tokenizer,
                                prompt_text=prompt,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                repetition_penalty=repetition_penalty,
                                max_new_tokens=cfg.max_new_tokens
                            )
                            if gen_tokens is None:
                                continue
                            gen_answer_list.append(trim_generation(gen_tokens))
                        for answer in gen_answer_list:
                            if qids[seq_id] not in prediced_results:
                                prediced_results[qids[seq_id]] = [answer]
                            else:
                                prediced_results[qids[seq_id]].append(answer)
                            result.append((question, answer))

                    ranked_predicted_results = collections.defaultdict(list)
                    sampled_answers = collections.defaultdict()
                    for qid in prediced_results:
                        counted_value = Counter(prediced_results[qid])
                        sampled_answers[qid] = counted_value
                        ranked_list = [pair[0] for pair in counted_value.most_common(10)]
                        ranked_predicted_results[qid] = ranked_list

                    with open(ranked_list_file, 'w') as fp:
                        for key in ranked_predicted_results:
                            json.dump({key: ranked_predicted_results[key]}, fp)
                            fp.write('\n')
                    with open(sample_answers_file, 'w') as fp:
                        json.dump(sampled_answers, fp)
                else:
                    logger.info(f"File {ranked_list_file} already exist")
        del model