import json
import pickle
from pathlib import Path

from omegaconf import DictConfig
import numpy as np

from calf import logger, OUTPUT
from calf.utils.log import progress_bar
from exps.corpus.blimp import all_blimp_dataset
from exps.utils import get_model_and_tokenizer, answer_probability_for_blimp


# -------------------- Files Functions --------------------
def get_blimp_root() -> Path:
    path = OUTPUT / "blimp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_probability_file_path(model_name: str) -> Path:
    path = get_blimp_root() / model_name / "probability"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_probability_file(model_name: str, blimp_topic: str) -> Path:
    path = get_probability_file_path(model_name)
    file = f"{blimp_topic}.pkl"
    return path / file


def get_accuracy_result_file() -> Path:
    path = get_blimp_root()
    file = f"blimp_arruracy_results.json"
    return path / file


# -------------------- Main Functions --------------------
def calc_blimp_probability(cfg: DictConfig) -> None:
    model_names = cfg.get("model", "llama2_13b")
    if isinstance(model_names, str):
        model_names = [model_names]

    blimp_dataset = all_blimp_dataset(cfg)
    for model_name in model_names:
        model, tokenizer = get_model_and_tokenizer(cfg[model_name], cfg.model_dtype, bos=True)
        for topic, dataset in blimp_dataset.items():
            hf_model_name = cfg[model_name].name.split('/')[-1]
            prob_file = get_probability_file(hf_model_name, topic)
            if not prob_file.is_file():
                results = []
                desc = f"Model: {model_name}; Topic: {topic}"
                for data in progress_bar(logger, dataset, desc=desc):
                    prob_good = answer_probability_for_blimp(
                        model=model,
                        tokenizer=tokenizer,
                        sentence=data["sentence_good"]
                    )
                    prob_bad = answer_probability_for_blimp(
                        model=model,
                        tokenizer=tokenizer,
                        sentence=data["sentence_bad"]
                    )
                    results.append({
                        "seqid": data["seqid"],
                        "blimp_data": data,
                        "probability_good": prob_good,
                        "probability_bad": prob_bad,
                    })
                with open(prob_file, mode="wb") as fp:
                    pickle.dump(results, fp)
                    logger.info(f"Save results to {prob_file}")
                del results
            else:
                logger.info(f"File {prob_file} already exist")
        del model


def calc_blimp_score(cfg: DictConfig) -> None:
    model_names = cfg.get("model", "llama2_13b")
    if isinstance(model_names, str):
        model_names = [model_names]

    blimp_dataset = all_blimp_dataset(cfg)
    result_file = get_accuracy_result_file()
    results = []
    for model_name in model_names:
        for i, (topic, dataset) in enumerate(blimp_dataset.items()):
            hf_model_name = cfg[model_name].name.split('/')[-1]
            prob_file = get_probability_file(hf_model_name, topic)
            if prob_file.is_file():
                with open(prob_file, mode="rb") as fp:
                    prob_results = pickle.load(fp)
                    correct = 0
                    total = 0
                    for result in prob_results:
                        p_good = np.sum(result["probability_good"])
                        p_bad = np.sum(result["probability_bad"])
                        correct = correct + 1 if p_good > p_bad else correct
                        total += 1
                    results.append({
                        "model_name": model_name,
                        "topic": topic,
                        "correct": correct,
                        "total": total
                    })
                logger.info(f"# {i} {prob_file.name}")
                logger.info(f"- model: {model_name}")
                logger.info(f"- blimp topic: {topic}")
                logger.info(f"- accuracy: {correct} / {total} = {correct / total}")
            else:
                raise FileNotFoundError(f"File {prob_file} not found")
    with open(result_file, mode='w', encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
