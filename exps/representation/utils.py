import json
import pickle
from omegaconf import DictConfig
from pathlib import Path
from typing import List, Tuple
from calf import logger, OUTPUT, CORPUS

from exps.concept.concept_inference import get_embedding_file, get_baseline_file

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import fasttext


def get_sorted_embs(cfg: DictConfig, emb_for_analysis: str = None) -> np.ndarray:
    if emb_for_analysis is None:
        emb_for_analysis = cfg.emb_for_analysis
    word_list, synset_list, emb_list = get_emb_info(cfg=cfg, emb_for_analysis=emb_for_analysis)
    sorted_embs = sort_emb_by_word_id(cfg=cfg, word_list=word_list, synset_list=synset_list, emb_list=emb_list)
    return sorted_embs


def get_emb_info(cfg: DictConfig, emb_for_analysis: str = None) -> Tuple[List, List,List]:
    # get embeddings from file
    embedding_file = get_embedding_file(
        cfg=cfg, model_name=cfg.embedding.model, task=cfg.embedding.task,
        corpus=cfg.embedding.dataset.corpus_name, n_demo=cfg.embedding.support_dataset.n_samples,
        n_query=cfg.embedding.query_dataset.n_samples, n_run=cfg.fid
    )
    if cfg.embs_to_use.lower().startswith("w2w") and not embedding_file.is_file():
        model = cfg.embedding.model.name.split('/')[-1]
        if cfg.embedding.model.get("revision", None):
            model = f"{model}_{cfg.embedding.model.revision}"
        suffix = "_".join(embedding_file.name.split("_")[1:]) if model != "phi-1_5" \
            else "_".join(embedding_file.name.split("_")[2:])
        embedding_file = embedding_file.parents[0] / f"{model}_None_{suffix}"
    if cfg.embs_to_use.lower().startswith("baseline"):
        embedding_file = get_baseline_file(
            cfg=cfg, model_name=cfg.embedding.model, task=cfg.embedding.task,
            corpus=cfg.embedding.dataset.corpus_name, n_demo=cfg.embedding.support_dataset.n_samples,
            n_query=cfg.embedding.query_dataset.n_samples, n_run=cfg.fid, baseline=cfg.baseline_type
        )
    assert embedding_file.is_file(), f"Check embedding_file at {embedding_file}"
    emb_list = []
    word_list = []
    synset_list = []
    with open(embedding_file, mode="rb") as fp:
        results = pickle.load(fp)
        logger.info(f"load from {embedding_file}")
        for i in range(len(results)):
            if emb_for_analysis is None:
                emb_for_analysis = cfg.emb_for_analysis
            if emb_for_analysis == "gen_embeddings":
                embs = [emb for w in results[i][emb_for_analysis] for emb in w]
                embs = np.stack(embs).astype(np.float64)
                assert len(embs.shape) == 2
                embs_for_ana = np.mean(embs, axis=0)
            else:
                embs_for_ana = results[i][emb_for_analysis]
            emb_list.append(embs_for_ana)
            word_list.append(results[i]["query_data"]["word"])
            synset_list.append(results[i]["query_data"]["synset"])
    return word_list, synset_list, emb_list


def sort_emb_by_word_id(cfg: DictConfig, word_list: List, synset_list: List, emb_list: List) -> np.ndarray:
    unique_id_file = Path(__file__).parent / "data" / "hmnrep" / cfg.unique_id_file
    assert unique_id_file.is_file(), f"Check unique_id_file at {unique_id_file}"
    # get word_synset_to_uid file
    ws2uid_file = Path(__file__).parent / "data" / cfg.ws2uid_file
    assert ws2uid_file.is_file(), f"Check unique_id_file at {ws2uid_file}"
    with open(ws2uid_file, mode="r") as f:
        ws2uid = json.load(f)

    # get unique word ids from word_list and synset_list
    uid_list = []
    for i in range(len(word_list)):
        ws = "_".join((word_list[i].replace(" ", "_"), synset_list[i]))
        uid_list.append(ws2uid[ws])

    # get sorted unique word ids
    with open(unique_id_file, mode="r") as f:
        lines = f.readlines()
    sorted_uids = []
    for i in range(len(lines)):
        sorted_uids.append(lines[i].strip("\n"))

    assert len(uid_list) == len(sorted_uids), \
        f"Mismatch between the embeddings and words: Embeddings {len(word_list)}, Words {len(lines)}"

    # get sorted embeddings
    sorted_emb_list = []
    for w in sorted_uids:
        assert w in uid_list, f"Cannot find word {w} in word_list from embedding_file"
        idx = uid_list.index(w)
        sorted_emb_list.append(emb_list[idx])
    sorted_embs = np.stack(sorted_emb_list, axis=0).astype(np.float64)

    return sorted_embs


def get_sorted_ft(cfg: DictConfig) -> np.ndarray:
    ft_file = Path(__file__).parent / "data" / "static_wv" / cfg.fasttext_file
    ft = fasttext.load_model(str(ft_file))
    things_file = CORPUS / "things" / cfg.things_corpus_file
    things_df = pd.read_csv(things_file, sep="\t")
    emb_list = []
    for i in range(len(things_df)):
        word = things_df.loc[i, "Word"]
        wv = ft.get_word_vector(word)
        emb_list.append(wv)
    sorted_ft = np.stack(emb_list, axis=0)
    return sorted_ft


def get_sorted_spose(cfg: DictConfig) -> np.ndarray:
    spose_embs_file = Path(__file__).parent / "data" / "hmnrep" / cfg.spose_embs_file
    sorted_embs = np.loadtxt(spose_embs_file)
    return sorted_embs


def cv_split_data(
        data: np.ndarray, n_splits: int = 10
) -> List[Tuple[np.ndarray, np.ndarray]]:
    kf = KFold(n_splits=n_splits, shuffle=True)
    train_test_indices = list(kf.split(data))
    return train_test_indices


def load_train_test_data(
        embs: np.ndarray, targets: np.ndarray, train_frac: float = 0.9
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]:
    emb_train_split, emb_test_split, d_idx = split_data(embs, train_frac=train_frac)
    target_train_split, target_test_split, _ = split_data(targets, train_frac=train_frac, d_idx=d_idx)
    return (emb_train_split, target_train_split), (emb_test_split, target_test_split), d_idx


def split_data(
        data: np.ndarray, train_frac: float = 0.9, d_idx: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if d_idx is None:
        num_samples = data.shape[0]
        d_idx = np.random.permutation(num_samples)
    train_split = data[d_idx[: int(len(d_idx) * train_frac)]]
    test_split = data[d_idx[int(len(d_idx) * train_frac):]]
    return train_split, test_split, d_idx
