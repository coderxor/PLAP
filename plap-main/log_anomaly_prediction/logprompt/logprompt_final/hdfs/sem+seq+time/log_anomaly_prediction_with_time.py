import os

import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
import matplotlib
matplotlib.use('Agg')  # 使用无头后端
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix
import torch.nn.functional as F


# 检查 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取和预处理数据
def load_and_process_data(log_file_path, label_file_path, window_duration, slide_interval,
                          next_window_duration, interval, n_rows):
    df_log = pd.read_csv(log_file_path,
                         usecols=['Date', 'Time', 'EventId', 'Content', 'EventTemplate', 'ParameterList'], nrows=n_rows)
    df_label = pd.read_csv(label_file_path)
    label_mapping = {'Normal': 0, 'Anomaly': 1}
    df_label['Label'] = df_label['Label'].map(label_mapping)

    label_dict = df_label.set_index('BlockId')['Label'].to_dict()

    def find_label(content):
        for block_id, label in label_dict.items():
            if block_id in content:
                return label
        return 0

    df_log['Label'] = df_log['Content'].apply(find_label)

    # 将 'Date' 列格式化为字符串，确保前导零保留
    df_log['Date'] = df_log['Date'].apply(lambda x: f"{x:06d}")

    # 合并 'Date' 和 'Time' 列
    df_log['DateTime_str'] = df_log['Date'].astype(str) + ' ' + df_log['Time'].astype(str)

    # 将合并后的字符串转换为 datetime 对象
    df_log['DateTime'] = pd.to_datetime(df_log['DateTime_str'], format='%y%m%d %H%M%S', errors='coerce')

    # 将 'DateTime' 转换为 Unix 时间戳（以秒为单位）
    df_log['Time_seconds'] = (df_log['DateTime'] - pd.Timestamp("1970-01-01")).dt.total_seconds()

    # 删除含有 NaT 的行
    df_log = df_log.dropna(subset=['Time_seconds'])
    print(df_log['Time_seconds'])

    # 排序数据按 'Time_seconds' 以便于后续操作
    df_log = df_log.sort_values(by='Time_seconds').reset_index(drop=True)

    texts = []
    labels = []

    start_time_seconds = df_log.iloc[0]['Time_seconds']
    while start_time_seconds < df_log['Time_seconds'].max() and len(texts) < 10000:
        # 当前窗口的结束时间
        end_time_seconds = start_time_seconds + window_duration

        # 被预测窗口的开始时间和结束时间
        next_window_start_time_seconds = end_time_seconds + interval
        next_window_end_time_seconds = next_window_start_time_seconds + next_window_duration

        # 检查被预测窗口是否超出了日志的时间范围，如果超出则停止处理
        if next_window_start_time_seconds >= df_log['Time_seconds'].max():
            print("没有足够的日志数据生成下一个被预测窗口，停止处理。")
            break

        # 选择窗口内的日志条目
        window_df = df_log[(df_log['Time_seconds'] >= start_time_seconds) & (df_log['Time_seconds'] < end_time_seconds)]

        # 选择被预测窗口内的日志条目
        next_window_df = df_log[(df_log['Time_seconds'] >= next_window_start_time_seconds) & (
                df_log['Time_seconds'] < next_window_end_time_seconds)]

        # print(f"Processing window from {start_time_seconds} to {end_time_seconds} with {len(window_df)} entries.")

        # 如果窗口内的日志条目为0，则跳过
        if len(window_df) == 0:
            start_time_seconds = end_time_seconds
            continue

        # 如果被预测窗口内的日志条目为0，则跳过
        if len(next_window_df) == 0:
            start_time_seconds = next_window_start_time_seconds
            continue

        combined_event_id = ""
        combined_content = ""
        combined_label = 0
        combined_times = []

        for _, row in window_df.iterrows():
            event_id = row['EventId'].split()[0]
            content = row['EventTemplate']
            # time = row['Time']

            combined_event_id += event_id + " "
            combined_content += " ".join(content.split()) + " "
            combined_label = max(combined_label, row['Label'])
            combined_times.append(row['Time_seconds'] - start_time_seconds)

        # 清理文本
        combined_event_id = combined_event_id.strip()
        combined_content = combined_content.strip()
        combined_content = re.sub(r'[^a-zA-Z0-9]', ' ', combined_content)

        # 生成文本格式
        combined_time = " ".join(map(str, combined_times))
        combined_text = f"semantic {combined_content} sequential {combined_event_id} time {combined_time}"
        combined_text = ' '.join(combined_text.split())

        texts.append(combined_text)

        # 为预测窗口分配被预测窗口的标签
        combined_label = max(next_window_df['Label'])
        labels.append(combined_label)

        # 更新窗口的开始时间为当前窗口的结束时间
        start_time_seconds = start_time_seconds + slide_interval
    # print(texts)
    return texts, labels


# HDFS
log_file_path = '/home/zhouzt/experiment/log_anomaly_prediction/logparser/Drain/demo_result/HDFS.log_structured.csv'
label_file_path = '/home/zhouzt/experiment/log_anomaly_prediction/logparser/Drain/demo_result/anomaly_label.csv'
texts, labels = load_and_process_data(log_file_path, label_file_path, window_duration=1,
                                      slide_interval=1, next_window_duration=1, interval=0, n_rows=1000000)
# interval为预测窗口和被预测窗口之间的间隔时间
# 打印texts的大小
print(f"Size of texts: {len(texts)}")

# 计算每个标签的数量
label_counts = pd.Series(labels).value_counts()
label_dict = label_counts.to_dict()

# 打印标签分布字典
print("原始数据集标签分布：")
print(label_dict)

# 将文本数据转换为DataFrame以便于使用imblearn进行处理
df = pd.DataFrame({'text': texts, 'label': labels})

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42,
                                                    stratify=df['label'])

# 打印训练集标签分布
train_label_counts = pd.Series(y_train).value_counts()
train_label_dict = train_label_counts.to_dict()

print("训练集标签分布:")
print(train_label_dict)

# 打印测试集标签数量
test_label_counts = pd.Series(y_test).value_counts()
test_label_dict = test_label_counts.to_dict()

print("测试集标签分布:")
print(test_label_dict)


# 定义降维层
class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)


# 处理嵌入和数据预处理
class LogDataset(Dataset):
    def __init__(self, texts, labels, max_length=512):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

        # 读取嵌入文件并解析
        self.embeddings_df = pd.read_csv("/home/zhouzt/experiment/log_anomaly_prediction/logprompt/logprompt_final/hdfs_id_template_embeddings.csv")
        self.embeddings_df['EventIdEmbedding'] = self.embeddings_df['EventIdEmbedding'].apply(self.parse_embedding)
        self.embeddings_df['EventTemplateEmbedding'] = self.embeddings_df['EventTemplateEmbedding'].apply(
            self.parse_embedding)

        # 初始化可训练的提示词嵌入
        self.semantic_keyword_embedding = nn.Parameter(torch.randn(300))  # 假设300维
        self.sequential_keyword_embedding = nn.Parameter(torch.randn(300))  # 假设300维
        self.time_keyword_embedding = nn.Parameter(torch.randn(300))  # 假设300维

        # 构建字典
        self.id_embeddings = self.embeddings_df.set_index('EventId')['EventIdEmbedding'].to_dict()
        self.template_embeddings = self.embeddings_df.set_index('EventId')['EventTemplateEmbedding'].to_dict()

        # 初始化
        self.projection_layer = ProjectionLayer(input_dim=768, output_dim=300)  # Word2Vec维度为300
        self.fc = nn.Linear(300 * 6, 768)  # 后续任务需要的向量维度
        # 假设有一个线性层将时间嵌入映射到300维
        self.time_embedding_layer = nn.Linear(1, 300)  # 输入为1维，输出为300维

    def parse_embedding(self, embedding_str):
        # 去掉字符串中的括号
        clean_str = embedding_str.replace('[', '').replace(']', '')
        # 将字符串分割并转换为浮点数
        embedding_list = list(map(float, clean_str.split()))
        return np.array(embedding_list)

    def time_to_embedding(self, relative_time):
        # 将相对时间转换为张量并调整形状
        relative_time_tensor = torch.tensor(relative_time, dtype=torch.float32)
        relative_time_tensor = relative_time_tensor.unsqueeze(0)  # 转换为 1D 张量

        # 使用线性层生成300维的嵌入
        time_embedding = self.time_embedding_layer(relative_time_tensor)  # 形状变为[300]
        return time_embedding

    def position_encoding(self, seq_len, embedding_dim):
        # 初始化位置嵌入矩阵
        position_embedding = torch.zeros(seq_len, embedding_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(np.log(10000.0) / embedding_dim))

        position_embedding[:, 0::2] = torch.sin(position * div_term)
        position_embedding[:, 1::2] = torch.cos(position * div_term)

        return position_embedding

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 使用正则表达式提取时间部分
        time_match = re.search(r'time (.+)$', text)
        if time_match:
            time_part = time_match.group(1).strip()
        else:
            time_part = ""

        # 提取ID部分
        split_text = text.split("sequential")
        if len(split_text) > 1:
            sequential_part = split_text[1].split("time")[0].strip()  # ID部分
        else:
            sequential_part = ""

        # 提取ID部分（按空格分割）
        event_ids = sequential_part.split()

        # 提取时间部分
        time_strings = time_part.split()
        relative_times = []
        for t in time_strings:
            try:
                relative_times.append(float(t))
            except ValueError:
                # 忽略无法转换为float的非数值内容
                continue

        # 模板和id嵌入
        event_id_embeddings = [self.id_embeddings.get(eid, np.zeros(768)) for eid in event_ids]
        template_embeddings = [self.template_embeddings.get(eid, np.zeros(768)) for eid in event_ids]

        # 创建最终嵌入列表
        final_embeddings = []

        # 位置编码
        s_len = 6  # 3个关键字 + 3个内容
        position_embeddings = self.position_encoding(s_len, 300)

        # 迭代每个事件ID
        for idx, eid in enumerate(event_ids):
            event_embedding = torch.tensor(event_id_embeddings[idx], dtype=torch.float32)
            template_embedding = torch.tensor(template_embeddings[idx], dtype=torch.float32)
            # 降维300
            event_embedding = self.projection_layer(event_embedding)
            template_embedding = self.projection_layer(template_embedding)

            # 获取相应的时间嵌入
            time_embedding = self.time_to_embedding(relative_times[idx])

            # 将位置编码应用到关键字和内容  每个相加后还是300维
            semantic_embedding_with_pos = self.semantic_keyword_embedding + position_embeddings[0]
            combined_content_with_pos = template_embedding + position_embeddings[1]

            sequential_embedding_with_pos = self.sequential_keyword_embedding + position_embeddings[2]
            combined_event_id_with_pos = event_embedding + position_embeddings[3]

            time_embedding_with_pos = self.time_keyword_embedding + position_embeddings[4]
            combined_time_with_pos = time_embedding + position_embeddings[5]

            # 拼接关键字和内容嵌入 600维
            final_semantic_embedding = torch.cat((semantic_embedding_with_pos, combined_content_with_pos), dim=-1)
            final_sequential_embedding = torch.cat((sequential_embedding_with_pos, combined_event_id_with_pos), dim=-1)
            final_time_embedding = torch.cat((time_embedding_with_pos, combined_time_with_pos), dim=-1)

            # 最终融合
            combined_embedding = torch.cat((final_semantic_embedding, final_sequential_embedding, final_time_embedding),
                                           dim=-1)
            final_embeddings.append(combined_embedding)

        # 转换为适合LSTM输入的格式
        final_embeddings = torch.stack(final_embeddings)  # shape: (seq_len, input_size) 3*600
        # 打印 final_embeddings 的形状
        # print("final_embeddings shape:", final_embeddings.shape)

        return final_embeddings, label


# 创建数据集
batch_size = 32
train_dataset = LogDataset(X_train.tolist(), y_train.tolist())
test_dataset = LogDataset(X_test.tolist(), y_test.tolist())


def collate_fn(batch):
    # 提取每个样本中的 final_embeddings 和 labels
    embeddings = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # 找到最大长度
    max_length = max(embedding.size(0) for embedding in embeddings)

    # 填充张量以使它们具有相同的大小
    padded_embeddings = []
    masks = []
    for embedding in embeddings:
        # 计算需要填充的大小
        padding_size = max_length - embedding.size(0)
        # 使用零进行填充
        padded_embedding = F.pad(embedding, (0, 0, 0, padding_size), 'constant', 0)
        padded_embeddings.append(padded_embedding)
        # 创建遮蔽：1 表示有效数据，0 表示填充
        mask = torch.ones(max_length, dtype=torch.float32)
        mask[-padding_size:] = 0  # 填充部分设为0
        masks.append(mask)

    # 将填充后的张量堆叠成一个批量张量
    embeddings_tensor = torch.stack(padded_embeddings)

    # 返回填充后的张量和标签
    return embeddings_tensor, labels


# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# LSTM + Attention 模型
class LSTMWithAttention(nn.Module):
    def __init__(self, input_size=3 * 600, hidden_size=128, num_layers=1, num_labels=2):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(p=0.3)

        # 可训练的注意力权重
        self.attention_weight_layer = nn.Linear(hidden_size, 1, bias=False)

    def attention(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_size)
        # 通过一个线性层计算注意力权重
        attn_weights = self.attention_weight_layer(lstm_output)  # (batch_size, seq_len, 1)

        # 将权重转换为概率分布
        attn_weights = nn.functional.softmax(attn_weights, dim=1)  # (batch_size, seq_len, 1)

        # 通过广播机制对 lstm_output 进行加权求和
        attn_output = torch.sum(attn_weights * lstm_output, dim=1)  # (batch_size, hidden_size)

        # 返回加权后的输出和权重
        return attn_output, attn_weights

    def forward(self, embeddings):
        # embeddings: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(embeddings)  # lstm_out: (batch_size, seq_len, hidden_size)

        # 使用自注意力机制
        attn_output, attn_weights = self.attention(lstm_out)

        # 将注意力机制输出的结果传入全连接层进行分类
        x = self.fc(self.dropout(attn_output))
        return x, attn_weights


# 损失函数Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# 计算类别权重
class_weights = compute_class_weight(class_weight={0: 1.0, 1: 1.0}, classes=np.array([0, 1]), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# 将模型移动到 GPU
model = LSTMWithAttention(input_size=3 * 600, hidden_size=128, num_layers=1, num_labels=2).to(device)

# 创建模型和优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-5)
criterion = FocalLoss(weight=class_weights.to(device))


# 训练函数
def train_model(model, criterion, optimizer, train_loader, num_epochs=10):
    best_f1 = 0
    model.train()
    train_losses = []
    train_accuracies = []
    train_recalls = []
    train_f1_scores = []
    train_precisions = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0
        all_train_labels = []
        all_train_predictions = []

        for embeddings, labels in train_loader:
            # 将数据移动到 GPU
            embeddings = embeddings.to(device)
            labels = torch.tensor(labels).to(device)

            optimizer.zero_grad()
            outputs, attn_weights = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * embeddings.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_predictions.extend(predicted.cpu().numpy())

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)
        train_recall = recall_score(all_train_labels, all_train_predictions, average='binary', pos_label=1)
        train_recalls.append(train_recall)
        train_f1_score = f1_score(all_train_labels, all_train_predictions, average='binary', pos_label=1)
        train_f1_scores.append(train_f1_score)
        train_precision = precision_score(all_train_labels, all_train_predictions, average='binary', pos_label=1,
                                          zero_division=0)
        train_precisions.append(train_precision)

        # 计算 TP, FP, FN
        tn, fp, fn, tp = confusion_matrix(all_train_labels, all_train_predictions).ravel()

        model.eval()

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Train Recall: {train_recall:.4f}, Train Precision: {train_precision:.4f}, "
              f"Train F1-score: {train_f1_score:.4f},"
              f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

        # 保存最优模型
        if train_f1_score > best_f1:
            best_f1 = train_f1_score
            torch.save(model.state_dict(), 'best_model1.pth')

    return train_losses, train_accuracies, train_recalls, train_precisions, train_f1_scores


# 训练模型
num_epochs = 50
train_losses, train_accuracies, train_recalls, train_precisions, train_f1_scores = train_model(model, criterion,
                                                                                               optimizer,
                                                                                               train_loader,
                                                                                               num_epochs)

# 绘制训练损失、准确率、精确率、召回率和F1-score曲线
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
plt.show()


# 加载最优模型进行评估
def evaluate_model(model, test_loader):
    model.load_state_dict(torch.load('best_model1.pth'))
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for embeddings, labels in test_loader:
            # 将数据移动到 GPU
            embeddings = embeddings.to(device)
            labels = torch.tensor(labels).to(device)

            outputs, attn_weights = model(embeddings)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    # 计算 TP, FP, FN
    # tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1:.4f}")
    # print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    # 删除模型文件

    if os.path.exists('best_model1.pth'):
        os.remove('best_model1.pth')  # 删除文件
        print("Model file 'best_model.pth' has been deleted.")


# 评估模型
evaluate_model(model, test_loader)
