from pathlib import Path
from omegaconf import OmegaConf

__version__ = "1.0.0"

ROOT = Path(__file__).parent.parent.resolve()

DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

CORPUS = DATA / "corpus"
CORPUS.mkdir(parents=True, exist_ok=True)

CACHE = DATA / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

# MODEL = ROOT / "models"
# MODEL.mkdir(parents=True, exist_ok=True)

HUGGINGFACE = ROOT.parent / "huggingface"
HUGGINGFACE.mkdir(parents=True, exist_ok=True)

OUTPUT = ROOT / "output"
OUTPUT.mkdir(parents=True, exist_ok=True)

device = "cpu"
logger = None
cfg = OmegaConf.create()
