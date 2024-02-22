from omegaconf import DictConfig
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np

from ..corpus import xcslb_dataset
from .utils import get_sorted_embs, get_sorted_ft, get_sorted_spose


CSLB_CONCEPT2UID = {
    "bat": "bat1",
    "bow": "bow2",
    "calf": "calf1",
    "camera": "camera1",
    "chicken": "chicken2",
    "mouse": "mouse1",
    "stove": "stove1",
    "tank": "tank1"
}


def pair_cslb_with_things(cfg: DictConfig) -> Tuple[np.ndarray, np.ndarray, List, Dict]:
    df, feat2count = xcslb_dataset()
    unique_id_file = Path(__file__).parent / "data" / "hmnrep" / cfg.unique_id_file
    assert unique_id_file.is_file() , f"Check unique_id_file at {unique_id_file}"

    # get sorted unique word ids
    with open(unique_id_file, mode="r") as f:
        lines = f.readlines()
    sorted_uids = []
    for i in range(len(lines)):
        sorted_uids.append(lines[i].strip("\n"))

    # map words in XCSLB dataset to unique word ids
    uid2w = dict()
    for c in df.concept:
        cid = c.replace(" ", "_")
        if cid in sorted_uids:
            uid2w[cid] = c
        else:
            assert cid in CSLB_CONCEPT2UID, f"Check unique word_id of {c}"
            uid2w[CSLB_CONCEPT2UID[cid]] = c
    embs_to_use = cfg.embs_to_use.lower()
    if embs_to_use == "model" or embs_to_use.startswith("baseline"):
        sorted_embs = get_sorted_embs(cfg=cfg)
    elif embs_to_use == "spose":
        sorted_embs = get_sorted_spose(cfg=cfg)
    elif embs_to_use == "fasttext":
        sorted_embs = get_sorted_ft(cfg=cfg)
    else:
        sorted_embs = get_sorted_embs(cfg=cfg)
    embs = []
    targets = []
    for i in range(len(sorted_uids)):
        uid = sorted_uids[i]
        if uid in uid2w:
            target = df[df.concept == uid2w[uid]].drop(columns=["concept"]).to_numpy().squeeze()
            targets.append(target)
            embs.append(sorted_embs[i])
    embs = np.stack(embs, axis=0)
    targets = np.stack(targets, axis=0)
    return embs, targets, df.drop(columns=["concept"]).columns.to_list(), feat2count

