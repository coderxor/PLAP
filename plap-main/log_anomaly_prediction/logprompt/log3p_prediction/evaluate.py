import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix

from log_anomaly_prediction.logprompt.log3p_prediction.config import *
from .utils.metrics import calculate_metrics


def evaluate_model(model, test_loader, model_path=None):
    """
    评估模型性能

    Args:
        model: 要评估的模型
        test_loader: 测试数据加载器
        model_path: 模型权重文件路径，如果为None则不加载模型

    Returns:
        评估指标
    """
    # 根据数据集选择模型文件
    if model_path is None:
        if DATASET == 'BGL':
            model_path = BGL_MODEL_SAVE_PATH
        elif DATASET == 'HDFS':
            model_path = HDFS_MODEL_SAVE_PATH
    
    # 加载模型（显式使用 weights_only + map_location，避免 FutureWarning 并提升可移植性）
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    
    model.eval()

    # 初始化指标统计
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    # 为BGL数据集使用tqdm进度条
    data_loader = tqdm(test_loader, desc="Evaluating") if DATASET == 'BGL' else test_loader

    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings = embeddings.to(DEVICE)
            labels = torch.tensor(labels).to(DEVICE)

            # 前向传播
            outputs, _ = model(embeddings)
            _, predicted = torch.max(outputs, 1)
            
            # 对于所有数据集，统计准确度
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
                
            # 收集标签和预测结果
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 统一计算评估指标，不再区分数据集
    accuracy = correct / total
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    
    # 计算混淆矩阵用于显示更详细的信息
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

    # 打印评估结果
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1-score: {metrics['f1']:.4f}")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}")

    return metrics