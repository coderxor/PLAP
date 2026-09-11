"""PLAP 统一配置：

阶段 1 日志解析 → Parsing
阶段 2 异常预测 → Prediction

"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 阶段 1：日志解析 (log_parsing / log3p_parsing)
# ---------------------------------------------------------------------------
class Parsing:
    RANDOM_SEED = 42
    CURRENT_DATASET = "bgl"  # hdfs / bgl / android / mac / zookeeper

    MODEL_OPTIONS = {
        "roberta-base": "/home/zhouzt/experiment/log_parsing/roberta-base",
        "roberta-large": "/home/zhouzt/experiment/log_parsing/roberta-large",
        "bert-base-uncased": "/home/zhouzt/experiment/log_parsing/bert-base-uncased",
        "bert-large-uncased": "/home/zhouzt/experiment/log_parsing/bert-large-uncased",
        "albert-base-v1": "/home/zhouzt/experiment/log_parsing/albert-base-v1",
        "albert-base-v2": "/home/zhouzt/experiment/log_parsing/albert-base-v2",
    }
    CURRENT_MODEL = "roberta-base"
    MODEL_NAME = MODEL_OPTIONS[CURRENT_MODEL]

    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 5e-5

    DATASETS = {
        "hdfs": {
            "file_path": "/home/zhouzt/experiment/log_parsing/logs/HDFS/HDFS_2k.log_structured.csv",
            # "file_path": "/home/zhouzt/experiment/log_anomaly_prediction/Drain_result/HDFS.log_structured.csv",
            # "file_path": "/home/zhouzt/experiment/log_anomaly_prediction/logparser/Drain/demo_result/HDFS_2k.log_structured.csv",
            "split_regex": r"[ ,!?=]+",
            "sample_count": 4,
        },
        "bgl": {
            "file_path": "/home/zhouzt/experiment/log_parsing/logs/BGL/BGL_2k.log_structured.csv",
            # "file_path": "/home/zhouzt/experiment/log_anomaly_prediction/Drain_result/BGL.log_structured.csv",
            # "file_path": "/home/zhouzt/experiment/log_anomaly_prediction/logparser/Drain/demo_result/BGL_2k.log_structured.csv",
            "split_regex": r"[ ,!?=]+",
            "sample_count": 4,
        },
        "android": {
            "file_path": "/home/zhouzt/experiment/log_parsing/logs/Android/Android_2k.log_structured.csv",
            # "file_path": "/home/zhouzt/experiment/log_parsing/30%/Android/selected_templates.csv",
            "split_regex": r"[ ,?_=]+",
            "sample_count": 4,
        },
        "mac": {
            "file_path": "/home/zhouzt/experiment/log_parsing/logs/Mac/Mac_2k.log_structured.csv",
            # "file_path": "/home/zhouzt/experiment/log_parsing/30%/Mac/selected_templates.csv",
            "split_regex": r"[ ,\[\]]+",
            "sample_count": 4,
        },
        "zookeeper": {
            "file_path": "/home/zhouzt/experiment/log_parsing/logs/Zookeeper/Zookeeper_2k.log_structured.csv",
            # "file_path": "/home/zhouzt/experiment/log_parsing/30%/Zookeeper/selected_templates.csv",
            "split_regex": r"[ .,!?]+",
            "sample_count": 4,
        },
    }

    TOKENS_TO_REMOVE = [
        "</s>", ".", "<s>", "]", "—", "[", ",", "...", ":", '"', ").", "-", "?",
        "(", "", "<unk>", ">", "<mask>", ")", "=", "+", "_", "/",
    ]
    TOP_K = 5  # 提取参数位置的候选词数量
    TOP_N = 8  # 选择频率最高的词数量

    EXPERIMENT = {
        "top_n_range": range(2, 53, 2),
        "datasets": ["hdfs", "bgl", "android", "mac", "zookeeper"],
        "metrics": ["pa", "ga"],
    }


def get_current_dataset_config():
    return Parsing.DATASETS[Parsing.CURRENT_DATASET]


# ---------------------------------------------------------------------------
# 阶段 2：异常预测 (log_anomaly_prediction / log3p_prediction)
# ---------------------------------------------------------------------------
class Prediction:
    RANDOM_SEED = 0
    TEST_SIZE = 0.2
    BATCH_SIZE = 1
    INTERVAL = 0
    N_ROWS = 800000

    BGL_MODEL_SAVE_PATH = "best_model_bgl.pth"
    HDFS_MODEL_SAVE_PATH = "best_model_hdfs.pth"

    # 提示词开关（实验有无提示词的影响）
    USE_PROMPT = True

    # 特征部分开关（实验不同特征组合）
    USE_SEMANTIC = True   # SEM
    USE_SEQUENTIAL = True  # SEQ
    USE_TIME = True        # TIME
    USE_PARAM = False      # PARAM

    WINDOW_DURATION = 8
    SLIDE_INTERVAL = 1
    NEXT_WINDOW_DURATION = 8

    DATASET = "HDFS"  # BGL 或 HDFS

    BGL_LOG_FILE_PATH = (
        "/home/zhouzt/experiment/log_anomaly_prediction/logparser/Drain/demo_result/BGL.log_structured_new.csv"
    )
    # BGL_EMBEDDING_FILE_PATH = "/home/zhouzt/experiment/log_anomaly_prediction/logprompt/BGL-120/bgl_id_template_embeddings.csv"
    BGL_EMBEDDING_FILE_PATH = (
        "/home/zhouzt/experiment/log_anomaly_prediction/logprompt/BGL-120/"
        "bgl_id_template_embeddings_bert_base_uncased.csv"
    )

    HDFS_LOG_FILE_PATH = (
        "/home/zhouzt/experiment/log_anomaly_prediction/logparser/Drain/demo_result/HDFS.log_structured_new.csv"
    )
    HDFS_LABEL_FILE_PATH = (
        "/home/zhouzt/experiment/log_anomaly_prediction/logparser/Drain/demo_result/anomaly_label.csv"
    )
    # HDFS_EMBEDDING_FILE_PATH = "/home/zhouzt/experiment/log_anomaly_prediction/logprompt/HDFS-56/hdfs_id_template_embeddings.csv"
    HDFS_EMBEDDING_FILE_PATH = (
        "/home/zhouzt/experiment/log_anomaly_prediction/logprompt/HDFS-56/"
        "hdfs_id_template_embeddings_bert_base_uncased.csv"
    )

    BGL_CONFIG = {
        "NUM_EPOCHS": 50,
        "EMBEDDING_DIM": 300,
        "HIDDEN_SIZE": 64,
        "NUM_LAYERS": 1,
        "NUM_LABELS": 2,
        "DROPOUT_RATE": 0.5,
        "LEARNING_RATE": 5e-5,
        "WEIGHT_DECAY": 6e-3,
        "FOCAL_LOSS_ALPHA": 0.25,
        "CLASS_WEIGHT_0": 1.0,
        "CLASS_WEIGHT_1": 1.0,
    }

    HDFS_CONFIG = {
        "NUM_EPOCHS": 50,
        "EMBEDDING_DIM": 300,
        "HIDDEN_SIZE": 128,
        "NUM_LAYERS": 1,
        "NUM_LABELS": 2,
        "DROPOUT_RATE": 0.5,
        "LEARNING_RATE": 5e-5,
        "WEIGHT_DECAY": 6e-3,
        "FOCAL_LOSS_ALPHA": 0.5,
        "CLASS_WEIGHT_0": 1.0,
        "CLASS_WEIGHT_1": 1.0,
    }


def sync_prediction():
    """按当前开关和 DATASET 展开派生项（EMBEDDING_PARTS、路径、超参）。"""
    p = Prediction
    active = []
    if p.USE_SEMANTIC:
        active.append("SEM")
    if p.USE_SEQUENTIAL:
        active.append("SEQ")
    if p.USE_TIME:
        active.append("TIME")
    if p.USE_PARAM:
        active.append("PARAM")
    p.ACTIVE_PARTS = active
    p.EMBEDDING_PARTS = len(active)
    if p.EMBEDDING_PARTS == 0:
        raise ValueError("至少需要启用一个特征部分 (USE_SEMANTIC/USE_SEQUENTIAL/USE_TIME/USE_PARAM)")

    if p.DATASET == "BGL":
        p.CONFIG = p.BGL_CONFIG
        p.LOG_FILE_PATH = p.BGL_LOG_FILE_PATH
        p.EMBEDDING_FILE_PATH = p.BGL_EMBEDDING_FILE_PATH
        p.LABEL_FILE_PATH = None
    elif p.DATASET == "HDFS":
        p.CONFIG = p.HDFS_CONFIG
        p.LOG_FILE_PATH = p.HDFS_LOG_FILE_PATH
        p.EMBEDDING_FILE_PATH = p.HDFS_EMBEDDING_FILE_PATH
        p.LABEL_FILE_PATH = p.HDFS_LABEL_FILE_PATH
    else:
        raise ValueError(f"未知数据集: {p.DATASET}")

    for key, value in p.CONFIG.items():
        setattr(p, key, value)


sync_prediction()
