import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt


def calculate_metrics(y_true, y_pred):
    """
    计算各种评估指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签

    Returns:
        包含各种指标的字典
    """
    accuracy = (y_pred == y_true).mean()
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def plot_training_curves(train_losses, train_accuracies, train_recalls,
                         train_precisions, train_f1_scores, num_epochs):
    """
    绘制训练曲线

    Args:
        train_losses: 训练损失列表
        train_accuracies: 训练准确率列表
        train_recalls: 训练召回率列表
        train_precisions: 训练精确率列表
        train_f1_scores: 训练F1分数列表
        num_epochs: 训练轮数
    """
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(range(1, num_epochs + 1), train_recalls, marker='o', label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall over Epochs')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(range(1, num_epochs + 1), train_precisions, marker='o', label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision over Epochs')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(range(1, num_epochs + 1), train_f1_scores, marker='o', label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.title('F1-score over Epochs')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()