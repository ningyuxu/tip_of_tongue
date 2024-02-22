from pathlib import Path
from typing import Tuple, Dict
import pandas as pd

from calf import CORPUS
from exps import init_config
from .things import all_things_dataset


def xcslb_dataset() -> Tuple[pd.DataFrame, Dict]:
    cfg = init_config(str(Path(__file__).parent), "config.yaml")
    file = CORPUS / "cslb" / cfg.cslb_corpus_file
    df = pd.read_csv(file)

    # remove rows for words not in THINGS
    things_words = [
        t["word"].lower() for t in all_things_dataset(fmt="wordnet")
    ]

    cslb_words = sorted(set(df.concept))
    overlap = []
    for w in cslb_words:
        if w.lower().replace('_', ' ') in things_words:
            overlap.append(w)
    df = df[df.concept.isin(overlap)]

    # remove features with extremely low density
    cslb_features = sorted(set(df.feature))
    feat2count = dict()
    for cf in cslb_features:
        feat2count[cf] = df.feature.value_counts()[cf].item()
    # sorted_feat_count = sorted(feat2count.items(), key=lambda x:x[1], reverse=True)
    keep_feats = []
    for feat in cslb_features:
        if feat2count[feat] >= cfg.cslb_feat_count_threshold:
            keep_feats.append(feat)
    df = df[df.feature.isin(keep_feats)]

    encoded = pd.get_dummies(df.feature)
    merged_df = pd.concat([df, encoded], axis='columns')
    merged_df = merged_df.drop(["feature", "label", "category"], axis='columns')
    df = merged_df.groupby('concept', as_index=False, sort=False).sum()
    return df, feat2count
