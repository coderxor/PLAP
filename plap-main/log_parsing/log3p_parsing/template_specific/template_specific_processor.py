import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TemplateSpecificDataset(Dataset):
    def __init__(self, dtrain, tokenizer, split_regex, template_to_param_id):
        self.dtrain = dtrain
        self.tokenizer = tokenizer
        self.vocab_set = set(tokenizer.get_vocab().keys())
        self.split_regex = split_regex
        self.template_to_param_id = template_to_param_id

    def __len__(self):
        return len(self.dtrain)

    def __getitem__(self, idx):
        log, template, line_ids, event_ids = self.dtrain[idx]

        # 使用配置的正则表达式分割
        input_tokens = re.split(self.split_regex, log)
        template_tokens = re.split(self.split_regex, template)

        param_indices = [i for i, token in enumerate(template_tokens) if token == "<*>"]

        # 创建目标序列
        target_tokens = input_tokens.copy()
        
        # 获取当前模板对应的特定PARAM标记
        template_id = event_ids
        param_token = f"PARAM{template_id}"
        
        # 将所有参数位置标记为模板特定的PARAM
        for idx in param_indices:
            if idx < len(target_tokens):
                target_tokens[idx] = param_token
            
        # 特殊处理连续参数的情况
        for i in range(len(param_indices) - 1):
            if param_indices[i] + 1 == param_indices[i+1]:
                if param_indices[i] < len(target_tokens) and param_indices[i+1] < len(target_tokens):
                    target_tokens[param_indices[i]] = param_token
                    target_tokens[param_indices[i+1]] = param_token

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)

        return log, input_ids, target_ids, template, line_ids, event_ids

def sample_logs_by_event(file_path, split_regex, n=1, random_state=42):
    """从结构化日志文件中按事件ID抽样日志"""
    structured_logs_df = pd.read_csv(file_path)
    grouped = structured_logs_df.groupby('EventId')
    Dtrain = []

    for event_id, group in grouped:
        # 抽样
        sampled_logs = group.sample(n=min(n, len(group)), random_state=random_state, replace=False)
        
        # 如果不够，继续抽样
        while len(sampled_logs) < n:
            additional_logs = group.sample(n=n - len(sampled_logs), random_state=random_state, replace=True)
            sampled_logs = pd.concat([sampled_logs, additional_logs], ignore_index=True)
        
        # 处理每条日志
        for idx, row in sampled_logs.iterrows():
            line_id = row['LineId']
            log = row['Content']
            event_template = row['EventTemplate']

            template_tokens = re.split(split_regex, event_template)
            
            # 保留原始的<*>标记
            processed_template = ['<*>' if '<*>' in token else token for token in template_tokens]
            template = ' '.join(processed_template)
            
            # 使用纯数字的模板ID（去掉'E'前缀）
            template_id = int(event_id[1:]) if event_id.startswith('E') else int(event_id)
            Dtrain.append((log, template, line_id, template_id))
    
    # 按模板ID排序
    Dtrain = sorted(Dtrain, key=lambda x: x[3])
    
    return Dtrain

def collate_fn(batch, device, tokenizer):
    """批处理函数，用于DataLoader"""
    logs, input_ids, target_ids, templates, line_ids, event_ids = zip(*batch)

    # 填充序列并移到设备
    input_ids = pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True,
                             padding_value=tokenizer.pad_token_id).to(device)
    target_ids = pad_sequence([torch.tensor(ids) for ids in target_ids], batch_first=True,
                              padding_value=tokenizer.pad_token_id).to(device)
    
    return logs, input_ids, target_ids, templates, line_ids, event_ids

def print_template_conversion_samples(dataset, split_regex, num_samples=None):
    """打印数据集中的模板转换示例，如果指定了数量则只打印指定数量的样本"""
    print("\n=== 日志模板转换结果 ===")
    num_to_print = len(dataset.dtrain) if num_samples is None else min(num_samples, len(dataset.dtrain))
    
    for idx in range(num_to_print):
        log, template, line_ids, event_ids = dataset.dtrain[idx]
        
        # 获取原始日志与模板令牌
        input_tokens = re.split(split_regex, log)
        template_tokens = re.split(split_regex, template)
        
        # 构建转换后的目标序列
        param_indices = [i for i, token in enumerate(template_tokens) if token == "<*>"]
        param_token = f"PARAM{event_ids}"
        
        # 基于模板结构创建目标序列
        target_tokens = template_tokens.copy()
        for i in param_indices:
            target_tokens[i] = param_token
        
        print(f"\n模板ID: {event_ids}")
        print(f"原始日志: {' '.join(input_tokens)}")
        print(f"模板: {' '.join(template_tokens)}")
        print(f"转换后: {' '.join(target_tokens)}")
        print("-" * 50)
    
    print("\n=== 转换结果打印完毕 ===\n")

def count_non_param_singleton_tokens(data, split_regex):
    """统计除参数位置外只出现一次的单词数量"""
    # 构建单词频率字典
    word_freq = {}
    param_tokens = set()  # 记录所有参数位置的单词
    
    # 第一遍：收集所有单词和参数位置的单词
    for log, template, _, _ in data:
        # 分析模板确定参数位置
        template_tokens = re.split(split_regex, template)
        log_tokens = re.split(split_regex, log)
        
        # 找到所有参数位置
        param_positions = [i for i, token in enumerate(template_tokens) if token == "<*>"]
        
        # 收集参数位置的单词
        for pos in param_positions:
            if pos < len(log_tokens):
                param_tokens.add(log_tokens[pos])
        
        # 统计所有非参数位置单词的频率
        for i, token in enumerate(log_tokens):
            if i not in param_positions:
                word_freq[token] = word_freq.get(token, 0) + 1
    
    # 计算只出现一次的非参数单词
    singleton_tokens = [token for token, freq in word_freq.items() if freq == 1]
    
    print(f"\n=== 日志统计信息 ===")
    print(f"总日志条数: {len(data)}")
    print(f"非参数位置的不同单词总数: {len(word_freq)}")
    print(f"非参数位置只出现一次的单词数: {len(singleton_tokens)}")
    print(f"参数位置单词总数: {len(param_tokens)}")
    
    # 打印部分只出现一次的单词示例
    sample_size = min(10, len(singleton_tokens))
    if sample_size > 0:
        print(f"\n只出现一次的非参数单词示例: {', '.join(singleton_tokens[:sample_size])}")
    
    return singleton_tokens

def create_template_specific_data_loaders(train_data, test_data, tokenizer, template_to_param_id, 
                                        batch_size, device, split_regex):
    """创建使用模板特定参数的训练和测试数据加载器"""
    # # 统计训练和测试数据中的单词信息
    # print("\n=== 训练数据统计 ===")
    # count_non_param_singleton_tokens(train_data, split_regex)
    #
    # print("\n=== 测试数据统计 ===")
    # count_non_param_singleton_tokens(test_data, split_regex)
    
    train_dataset = TemplateSpecificDataset(train_data, tokenizer, split_regex, template_to_param_id)
    test_dataset = TemplateSpecificDataset(test_data, tokenizer, split_regex, template_to_param_id)
    
    # 打印所有训练数据的模板转换结果
    print_template_conversion_samples(train_dataset, split_regex)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda x: collate_fn(x, device, tokenizer)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, device, tokenizer)
    )
    
    return train_loader, test_loader

def augment_template_specific_data(train_data, split_regex):
    """增强训练数据，特别关注连续参数的情况"""
    augmented_data = []
    
    for log, template, line_id, template_id in train_data:
        # 添加原始数据
        augmented_data.append((log, template, line_id, template_id))
        
        # 检查是否有连续参数
        template_tokens = re.split(split_regex, template)
        has_consecutive_params = False
        
        for i in range(len(template_tokens) - 1):
            if template_tokens[i] == "<*>" and template_tokens[i+1] == "<*>":
                has_consecutive_params = True
                break
        
        # 如果有连续参数，增加该样本的权重
        if has_consecutive_params:
            # 添加多个副本以增加权重
            for _ in range(2):
                augmented_data.append((log, template, line_id, template_id))
    
    return augmented_data 

