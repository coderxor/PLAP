import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class LogDataset(Dataset):
    def __init__(self, dtrain, tokenizer, split_regex, topk=5):
        self.dtrain = dtrain
        self.tokenizer = tokenizer
        self.vocab_set = set(tokenizer.get_vocab().keys())
        self.topk = topk
        self.split_regex = split_regex

    def __len__(self):
        return len(self.dtrain)

    def __getitem__(self, idx):
        log, template, line_ids, event_ids = self.dtrain[idx]

        # 确保log和template是字符串类型，处理None和NaN情况
        try:
            # 检查是否为NaN（包括numpy.nan, pandas.NA等）
            if log is None or (isinstance(log, float) and pd.isna(log)):
                log = ""
            elif not isinstance(log, str):
                log = str(log)
            # 确保转换后的字符串不为"nan"
            if log.lower() == "nan":
                log = ""
        except (TypeError, ValueError):
            log = ""
        
        try:
            if template is None or (isinstance(template, float) and pd.isna(template)):
                template = ""
            elif not isinstance(template, str):
                template = str(template)
            if template.lower() == "nan":
                template = ""
        except (TypeError, ValueError):
            template = ""

        # 使用配置的正则表达式分割
        input_tokens = re.split(self.split_regex, log)
        template_tokens = re.split(self.split_regex, template)

        param_indices = [i for i, token in enumerate(template_tokens) if token == "<*>"]

        # 创建目标序列
        target_tokens = input_tokens.copy()
        
        # 修改：确保每个参数位置都被正确标记为PARAM
        for idx in param_indices:
            if idx < len(target_tokens):
                target_tokens[idx] = "PARAM"
            
        # 添加：特殊处理连续参数的情况
        # 检查是否有连续的<*>，并确保对应位置都被标记为PARAM
        for i in range(len(param_indices) - 1):
            if param_indices[i] + 1 == param_indices[i+1]:
                # 连续参数情况
                if param_indices[i] < len(target_tokens) and param_indices[i+1] < len(target_tokens):
                    target_tokens[param_indices[i]] = "PARAM"
                    target_tokens[param_indices[i+1]] = "PARAM"

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
            try:
                line_id = row['LineId']
                log = row['Content']
                event_template = row['EventTemplate']
                
                # 检查必要字段是否为空或NaN
                if pd.isna(log) or pd.isna(event_template):
                    continue
                
                # 确保是字符串类型
                if log is None:
                    log = ""
                else:
                    log = str(log)
                    if log.lower() == "nan" or log == "None":
                        continue
                
                if event_template is None:
                    event_template = ""
                else:
                    event_template = str(event_template)
                    if event_template.lower() == "nan" or event_template == "None":
                        continue

                template_tokens = re.split(split_regex, event_template)
                            
                # 修改：保留原始的<*>标记，不要合并连续的<*>
                processed_template = ['<*>' if '<*>' in token else token for token in template_tokens]
                
                # 直接使用处理后的模板
                template = ' '.join(processed_template)
                Dtrain.append((log, template, line_id, event_id))
            except Exception as e:
                # 跳过有问题的行
                continue
    
    # 按 EventId 排序，确保相同EventId的日志放在一起
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

def create_data_loaders(train_data, test_data, tokenizer, batch_size, device, split_regex):
    """创建训练和测试数据加载器"""
    train_dataset = LogDataset(train_data, tokenizer, split_regex)
    test_dataset = LogDataset(test_data, tokenizer, split_regex)
    
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

def augment_data_for_consecutive_params(train_data, split_regex):
    """增强训练数据，特别关注连续参数的情况"""
    augmented_data = []
    
    for log, template, line_id, event_id in train_data:
        # 添加原始数据
        augmented_data.append((log, template, line_id, event_id))
        
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
            for _ in range(2):  # 增加2个副本，总共3份
                augmented_data.append((log, template, line_id, event_id))
    
    return augmented_data