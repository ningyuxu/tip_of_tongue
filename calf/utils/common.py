import random
import unicodedata
from typing import Dict, List

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
NUL = "<nul>"

MIN = -1e32
INF = float("inf")


def ispunct(token: str) -> bool:
    return all(unicodedata.category(char).startswith('P') for char in token)


def isfullwidth(token: str) -> bool:
    return all(
        unicodedata.east_asian_width(char) in ['W', 'F', 'A'] for char in token
    )


def islatin(token: str) -> bool:
    return all("LATIN" in unicodedata.name(char) for char in token)


def isdigit(token: str) -> bool:
    return all("DIGIT" in unicodedata.name(char) for char in token)


def tohalfwidth(token: str) -> str:
    return unicodedata.normalize("NFKC", token)


def get_signature(dataobject: Dict):
    import hashlib
    return hashlib.md5(str(list(dataobject.items())).encode()).hexdigest()


def partial_shuffle(x: List, d: float) -> List:
    """
    x: data to shuffle
    d: fraction of data to leave unshuffled
    """
    n = len(x)
    dn = int(d*n)
    indices = list(range(n))
    random.shuffle(indices)
    ind_fixed, ind_shuff = indices[dn:], indices[:dn]

    # copy across the fixed values
    result = x[:]

    # shuffle the shuffled values
    for src, dest in zip(ind_shuff, sorted(ind_shuff)):
        result[dest] = x[src]

    return result
