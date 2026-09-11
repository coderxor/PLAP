import torch
import numpy as np
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from log_anomaly_prediction.logprompt.log3p_prediction.config import *


class StaticFeatureMapper:
    """
    一个“确定性、可复用”的特征映射器：
    - 解决原实现中 train_dataset / test_dataset 各自随机初始化 projection/time/keyword/char embedding，
      导致训练与测试输入空间不一致，从而测试指标断崖式下降的问题。
    - 注意：它是静态映射（不参与训练），但至少保证 train/test 特征一致。
    """

    def __init__(self, seed: int = RANDOM_SEED):
        g = torch.Generator()
        g.manual_seed(int(seed))

        # 768 -> EMBEDDING_DIM 的线性投影（固定）
        self.proj_W = torch.randn(EMBEDDING_DIM, 768, generator=g) * 0.02
        self.proj_b = torch.zeros(EMBEDDING_DIM)

        # time: 1 -> EMBEDDING_DIM 的线性投影（固定）
        self.time_W = torch.randn(EMBEDDING_DIM, generator=g) * 0.02
        self.time_b = torch.zeros(EMBEDDING_DIM)

        # 4 个关键字 embedding（固定）
        self.semantic_kw = torch.randn(EMBEDDING_DIM, generator=g) * 0.02
        self.sequential_kw = torch.randn(EMBEDDING_DIM, generator=g) * 0.02
        self.time_kw = torch.randn(EMBEDDING_DIM, generator=g) * 0.02
        self.param_kw = torch.randn(EMBEDDING_DIM, generator=g) * 0.02

        # 字符级 embedding（固定）
        self.char_embedding = torch.randn(128, EMBEDDING_DIM, generator=g) * 0.02

    def project_768_to_d(self, x_768: torch.Tensor) -> torch.Tensor:
        # x_768: (768,)
        return torch.matmul(self.proj_W, x_768) + self.proj_b

    def time_to_embedding(self, relative_time: float) -> torch.Tensor:
        # relative_time: scalar -> (EMBEDDING_DIM,)
        t = torch.tensor(float(relative_time), dtype=torch.float32)
        return self.time_W * t + self.time_b


class LogDataset(Dataset):
    """日志数据集类，处理嵌入和特征提取"""

    def __init__(self, texts, labels, dataset_name=DATASET, max_length=512, feature_mapper: StaticFeatureMapper | None = None):
        """
        初始化数据集

        Args:
            texts: 文本列表
            labels: 标签列表
            dataset_name: 数据集名称，用于加载对应的模板参数向量
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.dataset_name = dataset_name.upper()
        self.feature_mapper = feature_mapper or StaticFeatureMapper()

        # 读取嵌入文件并解析
        self.load_embeddings()
        
        # 加载模板参数向量
        self.load_template_param_embeddings()
        
        # 根据启用的特征部分动态设置位置编码长度
        # 如果使用提示词：每个部分包含 keyword + content/time/param，所以是 2 * 启用部分数
        # 如果不使用提示词：每个部分只有 content/time/param，所以是 1 * 启用部分数
        self.active_parts = ACTIVE_PARTS
        if USE_PROMPT:
            self.s_len = len(self.active_parts) * 2
        else:
            self.s_len = len(self.active_parts) * 2  # 保持维度一致，但只使用一半的位置编码

    def load_embeddings(self):
        """加载和处理嵌入"""
        self.embeddings_df = pd.read_csv(EMBEDDING_FILE_PATH)
        self.embeddings_df['EventIdEmbedding'] = self.embeddings_df['EventIdEmbedding'].apply(self.parse_embedding)
        self.embeddings_df['EventTemplateEmbedding'] = self.embeddings_df['EventTemplateEmbedding'].apply(
            self.parse_embedding)

        # 构建字典
        self.id_embeddings = self.embeddings_df.set_index('EventId')['EventIdEmbedding'].to_dict()
        self.template_embeddings = self.embeddings_df.set_index('EventId')['EventTemplateEmbedding'].to_dict()
    
    def load_template_param_embeddings(self):
        """加载模板参数向量"""
        import pickle
        import os
        
        # 模板参数向量目录
        param_dir = f"/home/zhouzt/experiment/log_parsing/template_param/{self.dataset_name}/template_embeddings"
        
        # 存储模板参数向量的字典
        self.template_param_embeddings = {}
        
        # 如果目录存在
        if os.path.exists(param_dir):
            # 遍历目录中的所有文件
            for filename in os.listdir(param_dir):
                if filename.startswith("template_") and filename.endswith("_embedding.pkl"):
                    # 从文件名中提取模板ID
                    template_id_str = filename.replace("template_", "").replace("_embedding.pkl", "")
                    
                    # 加载参数向量
                    file_path = os.path.join(param_dir, filename)
                    with open(file_path, 'rb') as f:
                        param_embedding = pickle.load(f)
                        # 如果读取到的是numpy数组，就保持原样
                        self.template_param_embeddings[template_id_str] = param_embedding
            
            print(f"成功加载 {len(self.template_param_embeddings)} 个模板参数向量")
        else:
            print(f"警告: 模板参数向量目录 {param_dir} 不存在")

    def parse_embedding(self, embedding_str):
        """解析嵌入字符串为numpy数组"""
        clean_str = embedding_str.replace('[', '').replace(']', '')
        embedding_list = list(map(float, clean_str.split()))
        return np.array(embedding_list)

    def position_encoding(self, seq_len, embedding_dim):
        """生成位置编码"""
        position_embedding = torch.zeros(seq_len, embedding_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(np.log(10000.0) / embedding_dim))

        position_embedding[:, 0::2] = torch.sin(position * div_term)
        position_embedding[:, 1::2] = torch.cos(position * div_term)

        return position_embedding

    def get_char_level_param_embedding(self, param_str):
        """获取参数字符串的字符级别嵌入"""
        if not param_str:
            return torch.zeros(EMBEDDING_DIM)
            
        # 将字符串转换为ASCII码
        char_indices = torch.tensor([ord(c) for c in param_str if ord(c) < 128], dtype=torch.long)
        
        if len(char_indices) == 0:
            return torch.zeros(EMBEDDING_DIM)
            
        # 获取每个字符的嵌入
        char_embeddings = self.feature_mapper.char_embedding[char_indices]  # shape: (n_chars, embedding_dim)
        
        # 计算平均值
        avg_embedding = torch.mean(char_embeddings, dim=0)  # shape: (embedding_dim,)
        
        return avg_embedding

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 根据是否使用提示词采用不同的解析方式
        if USE_PROMPT:
            # 有提示词版本：使用关键字来分割
            time_match = re.search(r'time (.+) param', text)
            if time_match:
                time_part = time_match.group(1).strip()
            else:
                time_part = ""
                
            param_match = re.search(r'param (.+)$', text)
            if param_match:
                param_part = param_match.group(1).strip()
            else:
                param_part = ""
            
            split_text = text.split("sequential")
            if len(split_text) > 1:
                sequential_part = split_text[1].split("time")[0].strip()  # ID部分
            else:
                sequential_part = ""
        else:
            # 无提示词版本：需要从文本中提取四部分
            # 文本格式：{combined_content} {combined_event_id} {combined_time} {template_specific_param}
            # 策略：从后往前提取，因为后面的部分特征更明显
            
            parts = text.split()
            used_indices = set()  # 记录已使用的索引
            
            # 1. 先提取参数部分（template_开头，最明确，通常只有一个且在最后）
            param_part = ""
            for i, part in enumerate(parts):
                if part.startswith('template_'):
                    param_part = part
                    used_indices.add(i)
                    break
            
            # 2. 提取事件ID部分（E开头，后面跟数字，可能有多个）
            sequential_part = ""
            event_id_indices = []
            for i, part in enumerate(parts):
                if i in used_indices:
                    continue
                # 匹配事件ID格式：E后面跟数字（如E1, E2, E123等）
                if len(part) > 1 and part[0] == 'E' and part[1:].replace(' ', '').isdigit():
                    sequential_part += part + " "
                    event_id_indices.append(i)
                    used_indices.add(i)
            sequential_part = sequential_part.strip()
            
            # 3. 提取时间部分（纯数字，数量应该和事件ID一致）
            # 从后往前找数字，找到与事件ID数量一致的数字串
            time_part = ""
            num_event_ids = len(event_id_indices)
            if num_event_ids > 0:
                # 从后往前找数字（排除已使用的索引）
                time_values = []
                for i in range(len(parts) - 1, -1, -1):
                    if i in used_indices:
                        continue
                    try:
                        float(parts[i])
                        time_values.insert(0, parts[i])  # 保持顺序
                        used_indices.add(i)
                        if len(time_values) >= num_event_ids:
                            break
                    except ValueError:
                        pass
                
                # 确保时间值数量与事件ID数量一致
                if len(time_values) > num_event_ids:
                    # 如果找到太多，只取最后num_event_ids个
                    time_values = time_values[-num_event_ids:]
                elif len(time_values) < num_event_ids:
                    # 如果不够，用0填充
                    time_values = [str(0.0)] * (num_event_ids - len(time_values)) + time_values
                
                time_part = " ".join(time_values)
            
            # 4. 语义部分（combined_content）：剩下的所有部分
            # 注意：语义部分在这里不需要单独提取，因为后续会通过event_id来获取对应的template_embedding

        # 提取ID部分（按空格分割）
        event_ids = sequential_part.split()
        
        # 如果没有找到事件ID，添加一个默认ID避免后续处理出错
        if not event_ids:
            event_ids = ["E0"]  # 添加一个默认的事件ID
            print(f"警告: 文本未找到事件ID: {text}")

        # 提取时间部分
        time_strings = time_part.split()
        relative_times = []
        for t in time_strings:
            try:
                relative_times.append(float(t))
            except ValueError:
                # 忽略无法转换为float的非数值内容
                continue
        
        # 确保relative_times不为空，并且长度与event_ids一致
        if not relative_times:
            relative_times = [0.0] * len(event_ids)
        
        # 裁剪或填充相对时间以匹配事件ID数量
        if len(relative_times) < len(event_ids):
            relative_times.extend([0.0] * (len(event_ids) - len(relative_times)))
        elif len(relative_times) > len(event_ids):
            relative_times = relative_times[:len(event_ids)]

        # 模板和id嵌入
        event_id_embeddings = [self.id_embeddings.get(eid, np.zeros(768)) for eid in event_ids]
        template_embeddings = [self.template_embeddings.get(eid, np.zeros(768)) for eid in event_ids]
        
        # 获取模板参数嵌入
        param_embeddings = []
        for eid in event_ids:
            # 去掉'E'前缀获取纯数字模板ID
            template_id = eid[1:] if eid.startswith('E') else eid
            # 获取对应的模板参数向量，如果不存在则使用零向量
            param_embedding = self.template_param_embeddings.get(template_id, np.zeros(768))
            param_embeddings.append(param_embedding)

        # 修改位置编码
        position_embeddings = self.position_encoding(self.s_len, EMBEDDING_DIM)

        # 创建最终嵌入列表
        final_embeddings = []

        # 迭代每个事件ID
        for i, eid in enumerate(event_ids):
            # 防止索引错误
            if i >= len(event_id_embeddings) or i >= len(template_embeddings) or i >= len(param_embeddings) or i >= len(relative_times):
                print(f"警告: 索引 {i} 超出范围: event_ids={len(event_ids)}, embeddings={len(event_id_embeddings)}, times={len(relative_times)}")
                continue
            
            event_embedding = torch.tensor(event_id_embeddings[i], dtype=torch.float32)
            template_embedding = torch.tensor(template_embeddings[i], dtype=torch.float32)
            param_embedding = torch.tensor(param_embeddings[i], dtype=torch.float32)
            
            # 768 -> EMBEDDING_DIM（使用共享、确定性的 mapper）
            event_embedding = self.feature_mapper.project_768_to_d(event_embedding)
            template_embedding = self.feature_mapper.project_768_to_d(template_embedding)
            param_embedding = self.feature_mapper.project_768_to_d(param_embedding)

            # time -> EMBEDDING_DIM（共享、确定性）
            time_embedding = self.feature_mapper.time_to_embedding(relative_times[i])

            # 根据配置动态拼接特征部分
            parts_to_concat = []
            pos_idx = 0

            # SEM: 语义特征 (semantic keyword + template content)
            if 'SEM' in self.active_parts:
                if USE_PROMPT:
                    semantic_kw_with_pos = self.feature_mapper.semantic_kw + position_embeddings[pos_idx]
                    content_with_pos = template_embedding + position_embeddings[pos_idx + 1]
                    final_sem = torch.cat((semantic_kw_with_pos, content_with_pos), dim=-1)
                    pos_idx += 2
                else:
                    # 无提示词版本：只使用内容，不使用关键字
                    # 为了保持维度一致（2 * EMBEDDING_DIM），使用零向量作为关键字占位符
                    content_with_pos = template_embedding + position_embeddings[pos_idx]
                    zero_kw = torch.zeros(EMBEDDING_DIM)
                    final_sem = torch.cat((zero_kw, content_with_pos), dim=-1)
                    pos_idx += 2  # 仍然使用两个位置编码以保持一致性
                parts_to_concat.append(final_sem)

            # SEQ: 序列特征 (sequential keyword + event id)
            if 'SEQ' in self.active_parts:
                if USE_PROMPT:
                    sequential_kw_with_pos = self.feature_mapper.sequential_kw + position_embeddings[pos_idx]
                    event_id_with_pos = event_embedding + position_embeddings[pos_idx + 1]
                    final_seq = torch.cat((sequential_kw_with_pos, event_id_with_pos), dim=-1)
                    pos_idx += 2
                else:
                    # 无提示词版本：只使用事件ID，不使用关键字
                    event_id_with_pos = event_embedding + position_embeddings[pos_idx]
                    zero_kw = torch.zeros(EMBEDDING_DIM)
                    final_seq = torch.cat((zero_kw, event_id_with_pos), dim=-1)
                    pos_idx += 2
                parts_to_concat.append(final_seq)

            # TIME: 时间特征 (time keyword + time value)
            if 'TIME' in self.active_parts:
                if USE_PROMPT:
                    time_kw_with_pos = self.feature_mapper.time_kw + position_embeddings[pos_idx]
                    time_val_with_pos = time_embedding + position_embeddings[pos_idx + 1]
                    final_time = torch.cat((time_kw_with_pos, time_val_with_pos), dim=-1)
                    pos_idx += 2
                else:
                    # 无提示词版本：只使用时间值，不使用关键字
                    time_val_with_pos = time_embedding + position_embeddings[pos_idx]
                    zero_kw = torch.zeros(EMBEDDING_DIM)
                    final_time = torch.cat((zero_kw, time_val_with_pos), dim=-1)
                    pos_idx += 2
                parts_to_concat.append(final_time)

            # PARAM: 参数特征 (param keyword + param value)
            if 'PARAM' in self.active_parts:
                if USE_PROMPT:
                    param_kw_with_pos = self.feature_mapper.param_kw + position_embeddings[pos_idx]
                    param_val_with_pos = param_embedding + position_embeddings[pos_idx + 1]
                    final_param = torch.cat((param_kw_with_pos, param_val_with_pos), dim=-1)
                    pos_idx += 2
                else:
                    # 无提示词版本：只使用参数值，不使用关键字
                    param_val_with_pos = param_embedding + position_embeddings[pos_idx]
                    zero_kw = torch.zeros(EMBEDDING_DIM)
                    final_param = torch.cat((zero_kw, param_val_with_pos), dim=-1)
                    pos_idx += 2
                parts_to_concat.append(final_param)

            # 最终融合所有启用的特征部分
            if len(parts_to_concat) > 0:
                combined_embedding = torch.cat(parts_to_concat, dim=-1)
            else:
                # 如果所有特征都被禁用（不应该发生，但做保护）
                combined_embedding = torch.zeros(EMBEDDING_PARTS * 2 * EMBEDDING_DIM)
            
            final_embeddings.append(combined_embedding)

        # 确保final_embeddings不为空
        if not final_embeddings:
            # 创建一个全零的默认嵌入
            zero_embedding = torch.zeros(EMBEDDING_PARTS * 2 * EMBEDDING_DIM)  # EMBEDDING_PARTS个部分，每部分2个嵌入，每个嵌入EMBEDDING_DIM维
            final_embeddings.append(zero_embedding)
            print(f"警告: 样本 {idx} 未生成任何嵌入，使用零向量替代")

        # 转换为适合LSTM输入的格式
        final_embeddings = torch.stack(final_embeddings)  # shape: (seq_len, input_size) 4*600+300

        return final_embeddings, label


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
        if padding_size > 0:
            mask[-padding_size:] = 0  # 填充部分设为0
        masks.append(mask)

    # 将填充后的张量堆叠成一个批量张量
    embeddings_tensor = torch.stack(padded_embeddings)

    # 返回填充后的张量和标签
    return embeddings_tensor, labels


def create_data_loaders(X_train, X_test, y_train, y_test):
    """
    创建训练和测试数据加载器

    Args:
        X_train, X_test: 训练和测试文本
        y_train, y_test: 训练和测试标签

    Returns:
        训练和测试数据加载器
    """
    # 检查输入类型并转换为列表（如果需要）
    if hasattr(X_train, 'tolist'):
        X_train = X_train.tolist()
    if hasattr(y_train, 'tolist'):
        y_train = y_train.tolist()
    if hasattr(X_test, 'tolist'):
        X_test = X_test.tolist()
    if hasattr(y_test, 'tolist'):
        y_test = y_test.tolist()

    # 关键：train/test 必须共享同一套特征映射器，避免输入空间不一致
    shared_mapper = StaticFeatureMapper()
    train_dataset = LogDataset(X_train, y_train, feature_mapper=shared_mapper)
    test_dataset = LogDataset(X_test, y_test, feature_mapper=shared_mapper)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, test_loader
