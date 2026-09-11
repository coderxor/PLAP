"""兼容转发。改参数请编辑 plap-main/config.py 的 Prediction 段。"""
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
_root.sync_prediction()
Prediction = _root.Prediction
DEVICE = _root.DEVICE

RANDOM_SEED = Prediction.RANDOM_SEED
TEST_SIZE = Prediction.TEST_SIZE
BATCH_SIZE = Prediction.BATCH_SIZE
INTERVAL = Prediction.INTERVAL
N_ROWS = Prediction.N_ROWS
BGL_MODEL_SAVE_PATH = Prediction.BGL_MODEL_SAVE_PATH
HDFS_MODEL_SAVE_PATH = Prediction.HDFS_MODEL_SAVE_PATH
USE_PROMPT = Prediction.USE_PROMPT
USE_SEMANTIC = Prediction.USE_SEMANTIC
USE_SEQUENTIAL = Prediction.USE_SEQUENTIAL
USE_TIME = Prediction.USE_TIME
USE_PARAM = Prediction.USE_PARAM
ACTIVE_PARTS = Prediction.ACTIVE_PARTS
EMBEDDING_PARTS = Prediction.EMBEDDING_PARTS
WINDOW_DURATION = Prediction.WINDOW_DURATION
SLIDE_INTERVAL = Prediction.SLIDE_INTERVAL
NEXT_WINDOW_DURATION = Prediction.NEXT_WINDOW_DURATION
DATASET = Prediction.DATASET
BGL_LOG_FILE_PATH = Prediction.BGL_LOG_FILE_PATH
BGL_EMBEDDING_FILE_PATH = Prediction.BGL_EMBEDDING_FILE_PATH
HDFS_LOG_FILE_PATH = Prediction.HDFS_LOG_FILE_PATH
HDFS_LABEL_FILE_PATH = Prediction.HDFS_LABEL_FILE_PATH
HDFS_EMBEDDING_FILE_PATH = Prediction.HDFS_EMBEDDING_FILE_PATH
BGL_CONFIG = Prediction.BGL_CONFIG
HDFS_CONFIG = Prediction.HDFS_CONFIG
CONFIG = Prediction.CONFIG
LOG_FILE_PATH = Prediction.LOG_FILE_PATH
EMBEDDING_FILE_PATH = Prediction.EMBEDDING_FILE_PATH
LABEL_FILE_PATH = Prediction.LABEL_FILE_PATH
EMBEDDING_DIM = Prediction.EMBEDDING_DIM
HIDDEN_SIZE = Prediction.HIDDEN_SIZE
NUM_LAYERS = Prediction.NUM_LAYERS
NUM_LABELS = Prediction.NUM_LABELS
DROPOUT_RATE = Prediction.DROPOUT_RATE
LEARNING_RATE = Prediction.LEARNING_RATE
WEIGHT_DECAY = Prediction.WEIGHT_DECAY
NUM_EPOCHS = Prediction.NUM_EPOCHS
FOCAL_LOSS_ALPHA = Prediction.FOCAL_LOSS_ALPHA
CLASS_WEIGHT_0 = Prediction.CLASS_WEIGHT_0
CLASS_WEIGHT_1 = Prediction.CLASS_WEIGHT_1

print("启用长度：", EMBEDDING_PARTS)
