import torch
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
from sklearn.utils.class_weight import compute_class_weight

from log_anomaly_prediction.logprompt.log3p_prediction.config import *
from .utils.losses import FocalLoss
from .utils.metrics import plot_training_curves


def train_model(model, train_loader, num_epochs):
    """
    训练模型

    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        num_epochs: 训练轮数

    Returns:
        训练过程中的各种指标
    """
    # 根据配置决定是否使用balanced权重
    if CLASS_WEIGHT_0 == 'balanced' and CLASS_WEIGHT_1 == 'balanced':
        y_train = train_loader.dataset.labels
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        custom_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    else:
        custom_weights = torch.tensor([CLASS_WEIGHT_0, CLASS_WEIGHT_1], dtype=torch.float).to(DEVICE)
    
    # 设置优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=FOCAL_LOSS_ALPHA, weight=custom_weights)

    # 初始化指标记录
    best_f1 = 0
    train_losses = []
    train_accuracies = []
    train_recalls = []
    train_f1_scores = []
    train_precisions = []

    # 根据数据集选择不同的模型保存路径
    save_path = BGL_MODEL_SAVE_PATH if DATASET == 'BGL' else HDFS_MODEL_SAVE_PATH

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        all_train_labels = []
        all_train_predictions = []

        for embeddings, labels in train_loader:
            embeddings = embeddings.to(DEVICE)
            labels = torch.tensor(labels).to(DEVICE)

            # 前向传播
            optimizer.zero_grad()
            outputs, _ = model(embeddings)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 记录损失和预测
            epoch_train_loss += loss.item() * embeddings.size(0)
            _, predicted = torch.max(outputs, 1)
            
            # 统计准确率数据
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            # 收集标签和预测结果
            all_train_labels.extend(labels.cpu().numpy())
            all_train_predictions.extend(predicted.cpu().numpy())

        # 计算平均损失
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # 计算评估指标（统一逻辑，不再区分数据集）
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)
        train_recall = recall_score(all_train_labels, all_train_predictions, average='binary', pos_label=1)
        train_recalls.append(train_recall)
        train_f1_score = f1_score(all_train_labels, all_train_predictions, average='binary', pos_label=1)
        train_f1_scores.append(train_f1_score)
        train_precision = precision_score(all_train_labels, all_train_predictions, average='binary', pos_label=1,
                                        zero_division=0)
        train_precisions.append(train_precision)
        
        # 计算混淆矩阵元素
        tn, fp, fn, tp = confusion_matrix(all_train_labels, all_train_predictions).ravel()
        
        # 每个 epoch 只打一条日志
        saved = False
        if train_f1_score > best_f1:
            best_f1 = train_f1_score
            torch.save(model.state_dict(), save_path)
            saved = True

        print_str = (f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                    f"Train Recall: {train_recall:.4f}, Train Precision: {train_precision:.4f}, "
                    f"Train F1-score: {train_f1_score:.4f}, "
                    f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        if saved:
            print_str += f", Model saved (best F1: {best_f1:.4f})"
        print(print_str)

    # 绘制训练曲线
    plot_training_curves(
        train_losses, train_accuracies, train_recalls,
        train_precisions, train_f1_scores, epoch + 1
    )

    return train_losses, train_accuracies, train_recalls, train_precisions, train_f1_scores