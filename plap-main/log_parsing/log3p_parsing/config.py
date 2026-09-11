"""兼容转发。改参数请编辑 plap-main/config.py 的 Parsing 段。"""
import importlib.util
import sys
from pathlib import Path

_ROOT_NAME = "plap_root_config"


def _load_root():
    if _ROOT_NAME in sys.modules:
        return sys.modules[_ROOT_NAME]
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "config.py"
        if (
            candidate.is_file()
            and (parent / "log_parsing").is_dir()
            and (parent / "log_anomaly_prediction").is_dir()
        ):
            spec = importlib.util.spec_from_file_location(_ROOT_NAME, candidate)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[_ROOT_NAME] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("找不到 plap-main/config.py")


_root = _load_root()
Parsing = _root.Parsing
DEVICE = _root.DEVICE

RANDOM_SEED = Parsing.RANDOM_SEED
CURRENT_DATASET = Parsing.CURRENT_DATASET
MODEL_OPTIONS = Parsing.MODEL_OPTIONS
CURRENT_MODEL = Parsing.CURRENT_MODEL
MODEL_NAME = Parsing.MODEL_NAME
BATCH_SIZE = Parsing.BATCH_SIZE
EPOCHS = Parsing.EPOCHS
LEARNING_RATE = Parsing.LEARNING_RATE
DATASETS = Parsing.DATASETS
TOKENS_TO_REMOVE = Parsing.TOKENS_TO_REMOVE
TOP_K = Parsing.TOP_K
TOP_N = Parsing.TOP_N
EXPERIMENT = Parsing.EXPERIMENT


def get_current_dataset_config():
    return DATASETS[CURRENT_DATASET]
