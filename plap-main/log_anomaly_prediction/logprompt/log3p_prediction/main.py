import torch
import numpy as np
from sklearn.utils import compute_class_weight
import logging
import argparse
import matplotlib

matplotlib.use('Agg')  # 使用无头后端

from log_anomaly_prediction.logprompt.log3p_prediction.config import *
from .data.data_loader import load_and_process_data, split_dataset
from .data.dataset import create_data_loaders
from .models.lstm_attention import LSTMWithAttention
from .train import train_model
from .evaluate import evaluate_model


def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("log_anomaly_prediction.log"),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Log Anomaly Prediction')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--dataset', type=str, default=DATASET, choices=['BGL', 'HDFS'], help='Dataset to use')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--rows', type=int, default=N_ROWS, help='Number of rows to process')
    return parser.parse_args()


def main():
    """主函数"""
    # 设置日志
    setup_logging()

    # 解析参数
    args = parse_args()
    
    # 使用config.py中定义的随机种子固定随机性
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 更新数据集类型
    global DATASET, N_ROWS
    DATASET = args.dataset
    N_ROWS = args.rows if args.rows else N_ROWS
    
    # 根据数据集选择配置
    CONFIG = BGL_CONFIG if DATASET == 'BGL' else HDFS_CONFIG
    
    # 从配置中获取模型参数
    global EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, NUM_LABELS, DROPOUT_RATE
    global LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS
    
    EMBEDDING_DIM = CONFIG['EMBEDDING_DIM']
    HIDDEN_SIZE = CONFIG['HIDDEN_SIZE']
    NUM_LAYERS = CONFIG['NUM_LAYERS']
    NUM_LABELS = CONFIG['NUM_LABELS']
    DROPOUT_RATE = CONFIG['DROPOUT_RATE']
    
    # 使用命令行参数覆盖配置（如果提供）
    LEARNING_RATE = args.lr if args.lr else CONFIG['LEARNING_RATE']
    WEIGHT_DECAY = CONFIG['WEIGHT_DECAY']
    NUM_EPOCHS = args.epochs if args.epochs else CONFIG['NUM_EPOCHS']
    
    # 加载和处理数据
    logging.info(f"Loading and processing {DATASET} data...")
    texts, labels = load_and_process_data()

    # 分割数据集
    logging.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(texts, labels)

    # 创建数据加载器
    logging.info("Creating data loaders...")
    train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test)

    # 创建模型
    logging.info("Creating model...")
    input_size = EMBEDDING_PARTS * EMBEDDING_DIM * 2
    model = LSTMWithAttention(input_size=input_size).to(DEVICE)

    # 如果只需评估，跳过训练
    if not args.eval_only:
        # 训练模型
        logging.info("Training model...")
        train_losses, train_accuracies, train_recalls, train_precisions, train_f1_scores = train_model(
            model, 
            train_loader, 
            NUM_EPOCHS
        )

    # 评估模型
    logging.info("Evaluating model...")
    metrics = evaluate_model(model, test_loader)

    logging.info("Done!")


if __name__ == "__main__":
    main()