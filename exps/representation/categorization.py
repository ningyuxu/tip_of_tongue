import json
from typing import Optional
from omegaconf import DictConfig
from pathlib import Path
from calf import logger, OUTPUT

from exps import merge_config
from .utils import get_sorted_embs, get_sorted_ft, get_sorted_spose
from exps.concept.concept_inference import get_embedding_file

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn import metrics


THINGS_RMCAT = [2, 5, 7, 8, 11, 14, 25]
# remove:
# - bird 2
# - clothing accessory 5
# - dessert 7
# - drink 8
# - fruit 11
# - insect 14
# - vegetable 25


def categorize(cfg: DictConfig) -> None:
    cfg = merge_config(
        config_path=str(Path(__file__).parent),
        config_file="config.yaml",
        cfg=cfg
    )
    if cfg.embs_to_use.lower().startswith("baseline"):
        cfg.fid = 0
    if cfg.use_default_llms and cfg.embs_to_use.lower() not in ["fasttext", "spose"]:
        for llm in cfg.default_llms:
            cfg.embedding.model = llm
            if cfg.loop_over_fids:
                for fid in cfg.fids:
                    cfg.fid = int(fid)
                    run_categorize(cfg=cfg)
            else:
                run_categorize(cfg=cfg)
    else:
        run_categorize(cfg=cfg)


def run_categorize(cfg: DictConfig) -> None:
    if cfg.embs_to_use == "fasttext":
        sorted_embs = get_sorted_ft(cfg=cfg)
    elif cfg.embs_to_use == "spose":
        sorted_embs = get_sorted_spose(cfg=cfg)
    else:
        sorted_embs = get_sorted_embs(cfg=cfg)
    for dn in cfg.category_datasets:
        if cfg.embs_to_use.startswith("w2w"):
            result_dir = OUTPUT / cfg.save_dir / "w2w" / "categorization" / dn
        elif cfg.embs_to_use.startswith("desc") or cfg.emb_for_analysis.startswith("prompt_embedding"):
            result_dir = OUTPUT / cfg.save_dir / "desc" / "categorization" / dn
        else:
            result_dir = OUTPUT / cfg.save_dir / "categorization" / dn
        result_dir.mkdir(exist_ok=True, parents=True)
        file_name = "".join(str(get_embedding_file(
            cfg=cfg, model_name=cfg.embedding.model, task=cfg.embedding.task,
            corpus=cfg.embedding.dataset.corpus_name, n_demo=cfg.embedding.support_dataset.n_samples,
            n_query=cfg.embedding.query_dataset.n_samples, n_run=cfg.fid
        )).split("/")[-1].split(".")[:-1])
        if cfg.embs_to_use.lower() in ["fasttext", "spose"]:
            result_file = result_dir / f"{cfg.embs_to_use}_{cfg.dist_metric}_acc.json"
        elif cfg.embs_to_use.lower().startswith("baseline"):
            result_file = result_dir / f"{cfg.baseline_type}_{file_name}_{cfg.dist_metric}_acc.json"
        else:
            result_file = result_dir / f"{file_name}_{cfg.dist_metric}_acc.json"
        if result_file.is_file():
            logger.info(f">>> {result_file} exists. Skipping to next...")
        else:
            cat_mat = get_category_mat(cfg=cfg, dataset_name=dn, exclude_mul=True)
            embs = sorted_embs[cat_mat.index]
            labels = np.where(cat_mat.to_numpy() == 1)[1]

            num_samples = len(labels)
            uni_cats = sorted(set(labels))

            pred_labels = np.zeros_like(labels)
            results = {}

            for i in range(num_samples):
                test_idx = i
                train_idx = list(set(range(num_samples)).difference({i}))
                train_labels, test_label = labels[train_idx], labels[test_idx]
                x_train, x_test = embs[train_idx], embs[test_idx]

                cat_centroids = np.zeros((len(uni_cats), embs.shape[-1]))
                for c in range(len(uni_cats)):
                    x_cat = x_train[np.where(train_labels == uni_cats[c])[0]]
                    cat_centroids[c, :] = np.mean(x_cat, axis=0)

                if cfg.dist_metric == "dot":
                    dist2cent = np.dot(cat_centroids, x_test)
                    pred_labels[i] = np.argmax(dist2cent, axis=-1).item()
                else:
                    dist2cent = cdist(np.expand_dims(x_test, axis=0), cat_centroids, metric=cfg.dist_metric)
                    pred_labels[i] = np.argmax(-dist2cent, axis=-1).item()

            accuracy = np.mean(pred_labels == labels)
            balanced_acc = metrics.balanced_accuracy_score(labels, pred_labels)
            results["acc"] = accuracy
            results["balanced_acc"] = balanced_acc
            if cfg.embs_to_use in ["fasttext", "spose"]:
                logger.info(f"Accuracy of {cfg.embs_to_use} in categorization: {100 * accuracy}")
                            # f"Balanced accuracy: {balanced_acc}")
            elif cfg.embs_to_use.lower().startswith("baseline"):
                logger.info(f"Accuracy of {cfg.baseline_type}-{cfg.embedding.model.name.split('/')[1]} "
                            f"in categorization: {100 * accuracy}")
                            # f"Balanced accuracy: {balanced_acc}")
            else:
                logger.info(f"Accuracy of {cfg.embedding.model.name.split('/')[1]} in categorization: {100 * accuracy}")
                            # f"\nBalanced accuracy: {balanced_acc}")
            with open(result_file, mode="w") as outfile:
                json.dump(results, outfile)


def get_category_mat(cfg: DictConfig, dataset_name: str, exclude_mul: bool = True) -> pd.DataFrame:
    category_file = Path(__file__).parent / "data" / "hmnrep" / cfg.category_file_things
    # Remove categories that are subcategories of other categories
    if dataset_name == "things":
        cat_df = pd.read_csv(category_file)
        cat_df = cat_df.drop(columns=[f"Var{c}" for c in THINGS_RMCAT])
    else:
        raise ValueError(f"Unknown category dataset {dataset_name}")
    # Exclude objects that are part of multiple categories (unless subcategory)
    if exclude_mul:
        cat_df[cat_df.sum(axis=1) > 1] = 0
    # Remove categories with too few items (less than 10)
    rec = cat_df.sum(axis=0) < 10
    rm_cat = rec[rec].index.to_list()
    cat_df[rm_cat] = 0

    # Reduce to relevant objects
    cat_df = cat_df.loc[:, (cat_df != 0).any(axis=0)]
    cat_df = cat_df.loc[(cat_df != 0).any(axis=1), :]
    return cat_df
