from pathlib import Path
from typing import Dict

from omegaconf import DictConfig

from calf import CORPUS
from exps import merge_config


def all_sg_eval_dataset(cfg: DictConfig) -> Dict:
    cfg = merge_config(str(Path(__file__).parent), "config.yaml", cfg)
    pattern = cfg.sgeval_corpus_file_pattern
    corpus_path = CORPUS / "sg_eval"
    corpus_files = corpus_path.glob(pattern)
    if not corpus_files:
        raise FileNotFoundError(f"Files {pattern} not found")
    dataset_list = {}
    for file in corpus_files:
        dataset = []
        with open(file, mode='r', encoding="utf-8") as fp:
            for i, line in enumerate(fp):
                data = _parse_sg_eval_line(i, line.strip())
                dataset.append(data)
        dataset_list[file.stem] = dataset
    return dataset_list


def _parse_sg_eval_line(seqid: int, line: str) -> Dict:
    return {"line_id": seqid, "line": line}