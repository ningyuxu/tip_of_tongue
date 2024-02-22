import json
from pathlib import Path

from omegaconf import DictConfig
import numpy as np

from calf import logger, OUTPUT
from calf.utils.log import progress_bar
from exps.corpus import all_sg_eval_dataset
from exps.utils import get_model_and_tokenizer, answer_probability_for_sgeval


# -------------------- Files Functions --------------------
def get_sg_eval_root() -> Path:
    path = OUTPUT / "sg_eval"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_surprisal_file_path(model_name: str) -> Path:
    path = get_sg_eval_root() / model_name / "suprisal"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_surprisal_file(model_name: str, topic: str) -> Path:
    path = get_surprisal_file_path(model_name)
    file = f"{topic}.txt"
    return path / file


def get_accuracy_result_file() -> Path:
    path = get_sg_eval_root()
    file = f"sg_eval_arruracy_results.txt"
    return path / file


# -------------------- Main Functions --------------------
def calc_sg_surprisal(cfg: DictConfig) -> None:
    model_names = cfg.get("model", "llama2_13b")
    if isinstance(model_names, str):
        model_names = [model_names]

    sg_dataset = all_sg_eval_dataset(cfg)
    for model_name in model_names:
        model, tokenizer = get_model_and_tokenizer(cfg[model_name], cfg.model_dtype, bos=True)
        for topic, dataset in sg_dataset.items():
            hf_model_name = cfg[model_name].name.split('/')[-1]
            surprisal_file = get_surprisal_file(
                model_name=hf_model_name,
                topic=topic
            )
            if not surprisal_file.is_file():
                results = []
                desc = f"Model: {model_name}; Topic: {topic}"
                for data in progress_bar(logger, dataset, desc=desc):
                    probs = answer_probability_for_sgeval(
                        model=model,
                        tokenizer=tokenizer,
                        sentence=data["line"]
                    )
                    words = data["line"].split()
                    for i, (token, prob) in enumerate(zip(words, probs)):
                        results.append({
                            "sentence_id": data["line_id"] + 1,
                            "token_id": i + 1,
                            "token": token,
                            "surprisal": -np.sum(prob),
                        })
                with open(surprisal_file, mode='w', encoding="utf-8") as fp:
                    header = f"sentence_id\ttoken_id\ttoken\tsurprisal"
                    fp.write(f"{header}\n")
                    for r in results:
                        line = f"{r['sentence_id']}\t{r['token_id']}\t{r['token']}\t{r['surprisal']}"
                        fp.write(f"{line}\n")
                logger.info(f"Save results to {surprisal_file}")
                del results
            else:
                logger.info(f"File {surprisal_file} already exist")
        del model


def calc_sg_score(cfg: DictConfig) -> None:
    from exps.syntax.sg_eval.score import get_accuracy

    model_names = cfg.get("model", "llama2_13b")
    if isinstance(model_names, str):
        model_names = [model_names]

    sg_dataset = all_sg_eval_dataset(cfg)
    sg_eval_project_root = Path(__file__).parent / "sg_eval"

    results = []
    for model_name in model_names:
        hf_model_name = cfg[model_name].name.split('/')[-1]
        surprisal_path = get_surprisal_file_path(hf_model_name)
        for topic, _ in sg_dataset.items():
            input_file = sg_eval_project_root / "test_suites" / "json" / f"{topic}.json"
            surprisal = surprisal_path / f"{topic}.txt"
            sentences = sg_eval_project_root / "test_suites" / "txt" / f"{topic}.txt"
            spec = sg_eval_project_root / "spec.template.json"
            suite_name, accuracy = get_accuracy(
                input_path=input_file,
                surprisal_path=surprisal,
                sentence_path=sentences,
                spec_path=spec
            )
            results.append({
                "model": model_name,
                "topic": suite_name,
                "accuracy": accuracy,
            })
    result_file = get_accuracy_result_file()
    with open(result_file, mode='w', encoding="utf-8") as fp:
        json.dump(results, fp)