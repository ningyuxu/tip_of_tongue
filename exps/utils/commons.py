import re
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr


# noinspection PyTypeChecker
def rsa_score(X: np.ndarray, Y: np.ndarray, metric: str = "euclidean"):
    assert X.shape[0] == Y.shape[0]
    X_dist = squareform(pdist(X, metric=metric))
    Y_dist = squareform(pdist(Y, metric=metric))
    mask = np.zeros_like(X_dist)
    mask[np.triu_indices_from(mask)] = True
    np.fill_diagonal(mask, 0)
    half_x = X_dist[mask.astype(bool)]
    half_y = Y_dist[mask.astype(bool)]
    r, p_r = pearsonr(half_x, half_y)
    rho, p_rho = spearmanr(half_x, half_y)
    return {"pearsonr": {"r": r, "p": p_r}, "spearmanr": {"rho": rho, "p": p_rho}}


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
