import json

from omegaconf import DictConfig
from pathlib import Path
from typing import List, Dict
from calf import logger, OUTPUT

from exps import merge_config
from calf.utils.log import progress_bar
from .models import LogModel
from .data import pair_cslb_with_things
from .utils import cv_split_data, load_train_test_data
from exps.concept.concept_inference import get_embedding_file

import numpy as np
from sklearn.metrics import auc, roc_curve, f1_score, precision_score, accuracy_score
from scipy.special import expit


def regression(cfg: DictConfig) -> None:
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
            if cfg.loop_over_fids and not cfg.embs_to_use.lower().startswith("baseline"):
                for fid in cfg.fids:
                    cfg.fid = int(fid)
                    run_regression(cfg=cfg)
            else:
                run_regression(cfg=cfg)
    else:
        run_regression(cfg=cfg)


def run_regression(cfg: DictConfig) -> None:
    for fn in cfg.feature_names:
        if cfg.embs_to_use.startswith("w2w"):
            results_dir = OUTPUT / cfg.save_dir / "w2w" / "feature_ratings" / fn
        elif cfg.embs_to_use.startswith("desc") or cfg.emb_for_analysis.startswith("prompt_embedding"):
            results_dir = OUTPUT / cfg.save_dir / "desc" / "feature_ratings" / fn
        else:
            results_dir = OUTPUT / cfg.save_dir / "feature_ratings" / fn

        file_name = "".join(str(get_embedding_file(
            cfg=cfg, model_name=cfg.embedding.model, task=cfg.embedding.task,
            corpus=cfg.embedding.dataset.corpus_name, n_demo=cfg.embedding.support_dataset.n_samples,
            n_query=cfg.embedding.query_dataset.n_samples, n_run=cfg.fid
        )).split("/")[-1].split(".")[:-1])

        if not results_dir.is_dir():
            results_dir.mkdir(parents=True, exist_ok=True)
        if cfg.embs_to_use.lower() in ["spose", "fasttext"]:
            results_file = results_dir / f"{cfg.embs_to_use.lower()}_metrics.json"
        elif cfg.embs_to_use.lower().startswith("baseline"):
            results_file = results_dir / f"{cfg.baseline_type}_{file_name}_{cfg.embs_to_use.lower()}_metrics.json"
        else:
            results_file = results_dir / f"{file_name}_{cfg.embs_to_use.lower()}_metrics.json"
        if results_file.is_file():
            logger.info(f">>> {results_file} exists. Skipping to next...")
            continue

        if "cslb" in fn.lower():
            logger.info(f"Load data from {fn.upper()}")
            embs, targets, features, feat2count = pair_cslb_with_things(cfg=cfg)
            logistic_regression(cfg=cfg, feature_name=fn, embs=embs, targets=targets,
                                features=features, feat2count=feat2count, results_file=results_file,
                                results_dir=results_dir, file_name=file_name)
        else:
            raise ValueError(f"Unknown feature sets {fn}")


def logistic_regression(
        cfg: DictConfig, feature_name: str,
        embs: np.ndarray, targets: np.ndarray,
        features: List, feat2count: Dict,
        results_file: Path, results_dir: Path, file_name: str,
) -> None:
    if hasattr(cfg.reg, "cv") and cfg.reg.cv > 1:
        train_test_indices = cv_split_data(data=embs, n_splits=cfg.reg.cv)
        test_idx_records = []
        predict_records = []
        gold_records = []
        for i, (train_index, test_index) in progress_bar(logger, list(enumerate(train_test_indices))):

            emb_train, target_train = embs[train_index], targets[train_index]
            emb_test, target_test = embs[test_index], targets[test_index]
            test_idx_records.extend(test_index.tolist())

            model = LogModel()
            model.train(train_examples=emb_train, train_targets=target_train)

            if cfg.reg.save_params:
                if cfg.embs_to_use in ["fasttext", "spose"]:
                    param_file = results_dir / f"{cfg.embs_to_use.lower()}_reg_params_{i}.npz"
                else:
                    param_file = results_dir / f"{cfg.embs_to_use.lower()}_{file_name}_reg_params_{i}.npz"
                with open(param_file, "wb") as f:
                    np.savez_compressed(
                        f, weights=model.weights, intercepts=model.intercepts,
                        regularization_params=model.regularization_params,
                    )

            predict_targets = model.test(emb_test)
            predict_records.append(predict_targets)
            gold_records.append(target_test)

        predict_records = np.vstack(predict_records)
        gold_records = np.vstack(gold_records)

    else:
        if hasattr(cfg.reg, "cv") and cfg.reg.cv == 1:
            (emb_train, target_train), (emb_test, target_test), d_idx = load_train_test_data(
                embs, targets, train_frac=cfg.reg.train_frac
            )
        else:
            emb_train, target_train = embs, targets
            emb_test, target_test = embs, targets
            d_idx = np.arange(embs.shape[0]).tolist()

        model = LogModel()
        model.train(train_examples=emb_train, train_targets=target_train)

        if cfg.reg.save_params:
            if cfg.embs_to_use in ["fasttext", "spose"]:
                param_file = results_dir / f"{cfg.embs_to_use.lower()}_reg_params.npz"
            else:
                param_file = results_dir / f"{cfg.embs_to_use.lower()}_{file_name}_reg_params.npz"
            with open(param_file, "wb") as f:
                np.savez_compressed(
                    f, weights=model.weights, intercepts=model.intercepts,
                    regularization_params=model.regularization_params,
                )
        predict_targets = model.test(emb_test)
        predict_records = predict_targets
        gold_records = target_test
        test_idx_records = d_idx

    # Save prediction records
    if cfg.reg.save_predictions:
        with open(results_dir / f"{file_name}_{cfg.reg.prediction_file}", 'wb') as f:
            np.save(f, predict_records)

        with open(results_dir / f"{file_name}_{cfg.reg.gold_file}", 'wb') as f:
            np.save(f, gold_records)

        with open(results_dir / f"{file_name}_{cfg.reg.record_idx_file}", mode='w', encoding="utf-8") as f:
            print(json.dumps(test_idx_records), file=f)

    # Evaluation
    predict_probs = expit(predict_records)
    num_features = predict_records.shape[1]
    auc_records = []
    f1_records = []
    precision_records = []
    accuracy_records = []
    for i in range(num_features):
        pred = predict_probs[:, i]
        gold = gold_records[:, i]
        fpr, tpr, thresholds = roc_curve(gold, pred, pos_label=1)
        auc_records.append(auc(fpr, tpr).item())
        pred_label = (pred > 0.5).astype(int)
        f1 = f1_score(gold, pred_label, pos_label=1, average="binary")
        precision = precision_score(gold, pred_label, pos_label=1, average='binary')
        accuracy = accuracy_score(gold, pred_label)
        f1_records.append(f1.item())
        precision_records.append(precision.item())
        if isinstance(accuracy, np.float64):
            accuracy_records.append(accuracy.item())
        else:
            assert isinstance(accuracy, float)
            accuracy_records.append(accuracy)
    auc_mean = np.mean(auc_records).item()
    auc_median = np.median(auc_records).item()
    f1_mean = np.mean(f1_records).item()
    precision_mean = np.mean(precision_records).item()
    accuracy_mean = np.mean(accuracy_records).item()

    logger.info(
        f"Performance of {file_name} in feature ratings of {feature_name.upper()}: \n"
        f"AUC (Mean) = {auc_mean} \nAUC (Median) = {auc_median} \nF1 (Mean) = {f1_mean} \n"
        f"Precision(mean) = {precision_mean} \nAccuracy (mean) = {accuracy_mean} \n"
    )
    results = {
        "features": features, "feat2count": feat2count,
        "auc_mean": auc_mean, "auc_median": auc_median, "auc_records": auc_records,
        "f1_mean": f1_mean, "f1_records": f1_records,
        "precision_mean": precision_mean, "precision_records": precision_records,
        "accuracy_mean": accuracy_mean, "accuracy_records": accuracy_records,
    }

    with open(results_file, mode="w") as outfile:
        json.dump(results, outfile)
