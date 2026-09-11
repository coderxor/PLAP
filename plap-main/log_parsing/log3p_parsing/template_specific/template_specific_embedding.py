import re
import torch
import os
import json
import pickle
from collections import Counter, defaultdict
from transformers import AutoModelForMaskedLM, AutoTokenizer
from log_parsing.log3p_parsing.config import DEVICE, TOKENS_TO_REMOVE, TOP_K, TOP_N

def add_new_tokens(tokens, tokenizer, model):
    """动态扩充词汇表并调整模型嵌入"""
    new_tokens = [token for token in tokens if token not in tokenizer.get_vocab()]

    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)

    return new_tokens

def get_token_probabilities(model, tokenizer, text, template, split_regex, device):
    """利用预训练语言模型获取概率分布"""
    tokens = re.split(split_regex, text)
    template_tokens = re.split(split_regex, template)
    
    # 找到参数位置
    param_positions = [i for i, token in enumerate(template_tokens) if token == "<*>"]
    
    add_new_tokens(tokens, tokenizer, model)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = torch.ones(1, len(input_ids), device=device)  # 直接在目标设备上创建
    
    # 创建参数位置的掩码，直接在目标设备上
    param_mask = torch.ones_like(attention_mask, device=device)
    # for pos in param_positions:
    #     if pos < param_mask.size(1):
    #         param_mask[0, pos] = 0
    
    inputs = {
        'input_ids': torch.tensor([input_ids], device=device),  # 直接在目标设备上创建
        'attention_mask': attention_mask,
        'param_attention_mask': param_mask
    }
    
    with torch.no_grad():
        # 在forward时使用参数掩码
        outputs = model(**{k: v for k, v in inputs.items() if k != 'param_attention_mask'})
        
        # 对参数位置的logits进行掩码处理
        logits = outputs.logits
        param_mask_expanded = param_mask.unsqueeze(-1).expand_as(logits)
        masked_logits = logits * param_mask_expanded
        
        # 计算概率分布
        probabilities = torch.softmax(masked_logits, dim=-1)
    
    return probabilities, inputs

def extract_param_tokens(text, template, tokenizer, probabilities, inputs, split_regex, device, top_k=TOP_K):
    """提取参数位置并获取Top-k预测Token"""
    tokens_in_template = re.split(split_regex, template)
    param_positions = [i for i, token in enumerate(tokens_in_template) if token == "<*>"]

    if not param_positions:
        return {}
    
    top_k_tokens = {}
    for pos in param_positions:
        if pos < probabilities.size(1):
            token_probs = probabilities[0, pos].to(device)
            top_k_indices = torch.topk(token_probs, top_k).indices
            top_k_tokens[pos] = [tokenizer.decode([idx]).strip() for idx in top_k_indices]
    
    return top_k_tokens

def generate_template_specific_v_ini(template_data):
    """为每个模板生成参数指示集合"""
    template_v_ini = {}
    for template_id, top_k_tokens_list in template_data.items():
        v_ini = []
        for top_k_tokens in top_k_tokens_list:
            for tokens in top_k_tokens.values():
                v_ini.extend(tokens)
        template_v_ini[template_id] = Counter(v_ini)
    return template_v_ini

def select_pivot_tokens_per_template(template_v_ini, tokens_to_remove=TOKENS_TO_REMOVE, top_n=None):
    """为每个模板选择频率最高的Token"""
    if top_n is None:
        top_n = TOP_N
        
    template_pivot_tokens = {}
    
    print("\n=== 每个模板的Top-{}高频Token ===".format(top_n))
    for template_id, v_ini in template_v_ini.items():
        filtered_tokens = []
        print(f"\n模板 {template_id} 的高频Token:")
        for token, freq in v_ini.most_common():
            print(f"    Token: {token:<20} 频率: {freq}")
            filtered_tokens.append(token)
            if len(filtered_tokens) == top_n:
                break
        template_pivot_tokens[template_id] = filtered_tokens
    print("\n===============================")
    
    return template_pivot_tokens

def get_template_specific_embeddings(model, tokenizer, template_pivot_tokens, device):
    """计算每个模板特定的参数嵌入向量"""
    template_embeddings = {}
    
    for template_id, pivot_tokens in template_pivot_tokens.items():
        embeddings = []
        for token in pivot_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            # 根据模型类型获取词嵌入
            if hasattr(model, 'roberta'):
                embedding = model.roberta.embeddings.word_embeddings.weight[token_id].to(device)
            elif hasattr(model, 'bert'):
                embedding = model.bert.embeddings.word_embeddings.weight[token_id].to(device)
            elif hasattr(model, 'albert'):
                embedding = model.albert.embeddings.word_embeddings.weight[token_id].to(device)
            else:
                raise ValueError("不支持的模型类型，当前仅支持RoBERTa、BERT和ALBERT模型")
            embeddings.append(embedding)
        
        if embeddings:
            virtual_embedding = torch.mean(torch.stack(embeddings), dim=0)
            template_embeddings[template_id] = virtual_embedding
    
    return template_embeddings

def add_template_specific_embeddings(model, tokenizer, template_embeddings, device):
    """将模板特定的参数嵌入添加到模型中"""
    # 创建参数token名称列表
    param_tokens = [f"PARAM{i+1}" for i in range(len(template_embeddings))]
    
    # 准备所有虚拟嵌入向量
    all_embeddings = torch.stack([emb.to(device) for emb in template_embeddings.values()])
    
    # 根据模型类型更新词嵌入
    if hasattr(model, 'roberta'):
        new_weight = torch.cat([model.roberta.embeddings.word_embeddings.weight, all_embeddings])
        model.roberta.embeddings.word_embeddings.weight = torch.nn.Parameter(new_weight)
    elif hasattr(model, 'bert'):
        new_weight = torch.cat([model.bert.embeddings.word_embeddings.weight, all_embeddings])
        model.bert.embeddings.word_embeddings.weight = torch.nn.Parameter(new_weight)
    elif hasattr(model, 'albert'):
        new_weight = torch.cat([model.albert.embeddings.word_embeddings.weight, all_embeddings])
        model.albert.embeddings.word_embeddings.weight = torch.nn.Parameter(new_weight)
    else:
        raise ValueError("不支持的模型类型")
    
    # 添加新的PARAM tokens到分词器
    tokenizer.add_tokens(param_tokens)
    
    # 更新模型大小以匹配新的词汇表大小
    model.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = len(tokenizer)
    
    # 创建模板ID到对应PARAM token ID的映射
    template_to_param_id = {}
    for i, template_id in enumerate(template_embeddings.keys()):
        param_token = param_tokens[i]
        param_id = tokenizer.convert_tokens_to_ids(param_token)
        template_to_param_id[template_id] = param_id
    
    return model, tokenizer, template_to_param_id

def save_template_info(dataset_name, template_data, template_v_ini, template_pivot_tokens,
                      template_embeddings, template_to_param_id, template_to_event_id,
                      save_dir="/home/zhouzt/experiment/log_parsing/template_param"):
    """保存模板参数信息到指定目录，按数据集组织"""
    # 创建数据集专属目录
    dataset_dir = os.path.join(save_dir, dataset_name.upper())
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 保存模板数量信息
    template_count = len(template_data)
    print(f"\n数据集 {dataset_name} 共有 {template_count} 个模板")
    
    # 保存模板原始数据 (JSON格式)
    with open(os.path.join(dataset_dir, 'template_data.json'), 'w') as f:
        # 将字典的键转换为字符串，以便JSON序列化
        serializable_data = {}
        for template_id, data in template_data.items():
            event_id = template_to_event_id.get(template_id, template_id)
            serializable_data[str(event_id)] = []
            for item in data:
                # 转换内部字典的键为字符串
                converted_item = {}
                for pos, tokens in item.items():
                    converted_item[str(pos)] = tokens
                serializable_data[str(event_id)].append(converted_item)
        json.dump(serializable_data, f, indent=2)
    
    # 保存参数指示词集合 (JSON格式)
    with open(os.path.join(dataset_dir, 'template_v_ini.json'), 'w') as f:
        # Counter不能直接序列化，转换为普通字典
        serializable_v_ini = {}
        for template_id, counter in template_v_ini.items():
            event_id = template_to_event_id.get(template_id, template_id)
            serializable_v_ini[str(event_id)] = dict(counter)
        json.dump(serializable_v_ini, f, indent=2)
    
    # 保存模板高频词 (JSON格式)
    with open(os.path.join(dataset_dir, 'template_pivot_tokens.json'), 'w') as f:
        # 转换键为字符串，使用event_id作为键
        serializable_pivot_tokens = {}
        for template_id, tokens in template_pivot_tokens.items():
            event_id = template_to_event_id.get(template_id, template_id)
            serializable_pivot_tokens[str(event_id)] = tokens
        json.dump(serializable_pivot_tokens, f, indent=2)
    
    # 为每个模板单独保存嵌入向量
    template_embeddings_dir = os.path.join(dataset_dir, 'template_embeddings')
    os.makedirs(template_embeddings_dir, exist_ok=True)
    
    # 同时创建一个映射文件，记录维度信息
    embedding_info = {}
    
    for template_id, embedding in template_embeddings.items():
        # 获取事件ID
        event_id = template_to_event_id.get(template_id, template_id)
        
        # 分离梯度并转换为numpy数组
        embedding_np = embedding.detach().cpu().numpy()
        
        # 记录维度信息
        embedding_info[str(event_id)] = {
            "shape": list(embedding_np.shape),
            "template_id": int(template_id),
            "event_id": event_id,
            "file_name": f"template_{event_id}_embedding.pkl"
        }
        
        # 保存单个模板的嵌入向量
        with open(os.path.join(template_embeddings_dir, f"template_{event_id}_embedding.pkl"), 'wb') as f:
            pickle.dump(embedding_np, f)
    
    # 保存嵌入向量信息索引
    with open(os.path.join(dataset_dir, 'embedding_info.json'), 'w') as f:
        json.dump(embedding_info, f, indent=2)
    
    # 保存模板ID到参数ID的映射 (JSON格式)
    with open(os.path.join(dataset_dir, 'template_to_param_id.json'), 'w') as f:
        # 转换键为字符串，使用event_id作为键
        serializable_mapping = {}
        for template_id, param_id in template_to_param_id.items():
            event_id = template_to_event_id.get(template_id, template_id)
            serializable_mapping[str(event_id)] = param_id
        json.dump(serializable_mapping, f, indent=2)
    
    # 保存模板参数概述 (TXT格式)
    with open(os.path.join(dataset_dir, 'template_summary.txt'), 'w') as f:
        f.write(f"数据集: {dataset_name.upper()}\n")
        f.write(f"模板总数: {template_count}\n\n")
        
        for template_id in template_data.keys():
            event_id = template_to_event_id.get(template_id, template_id)
            f.write(f"模板ID: {template_id}\n")
            f.write(f"事件ID: {event_id}\n")
            f.write(f"参数Token: PARAM{template_id}\n")
            
            # 写入该模板的高频词
            if str(event_id) in serializable_pivot_tokens:
                f.write("高频Token:\n")
                for token in serializable_pivot_tokens[str(event_id)]:
                    f.write(f"  - {token}\n")
            
            f.write("\n" + "-"*50 + "\n\n")
    
    print(f"\n数据集 {dataset_name.upper()} 的模板参数信息已保存到: {dataset_dir}")
    print(f"共有 {template_count} 个模板的参数向量被单独保存到: {template_embeddings_dir}")

def create_template_specific_embeddings(train_data, split_regex, model_name, dataset_name=None):
    """创建模板特定嵌入的主函数"""
    # 如果未提供数据集名称，尝试从模型名称中提取
    if dataset_name is None:
        # 尝试从模型名称中提取数据集名称
        if '/' in model_name:
            dataset_name = model_name.split('/')[-1]
        else:
            dataset_name = "unknown_dataset"
    
    # 加载预训练模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )
    model = model.to(DEVICE)
    
    # 获取并打印嵌入维度
    if hasattr(model, 'roberta'):
        embedding_dim = model.roberta.embeddings.word_embeddings.weight.shape[1]
    elif hasattr(model, 'bert'):
        embedding_dim = model.bert.embeddings.word_embeddings.weight.shape[1]
    elif hasattr(model, 'albert'):
        embedding_dim = model.albert.embeddings.word_embeddings.weight.shape[1]
    else:
        embedding_dim = "未知"
    
    print(f"\n使用模型: {model_name}")
    print(f"参数向量维度: {embedding_dim}")
    
    # 按模板ID组织数据
    template_data = defaultdict(list)
    template_to_event_id = {}
    
    for log, template, line_id, event_id in train_data:
        # 假设event_id就是模板ID
        template_id = event_id
        template_to_event_id[template_id] = event_id
        
        probabilities, inputs = get_token_probabilities(model, tokenizer, log, template, split_regex, DEVICE)
        top_k_tokens = extract_param_tokens(log, template, tokenizer, probabilities, inputs, split_regex, DEVICE)
        template_data[template_id].append(top_k_tokens)
    
    # 为每个模板生成参数指示词集合
    template_v_ini = generate_template_specific_v_ini(template_data)
    
    # 为每个模板选择最频繁的tokens
    template_pivot_tokens = select_pivot_tokens_per_template(template_v_ini)
    
    # 计算每个模板的参数嵌入向量
    template_embeddings = get_template_specific_embeddings(model, tokenizer, template_pivot_tokens, DEVICE)
    
    # 添加模板特定的参数嵌入
    updated_model, updated_tokenizer, template_to_param_id = add_template_specific_embeddings(
        model, tokenizer, template_embeddings, DEVICE
    )
    
    # 保存模板参数信息，并传入template_to_event_id
    save_template_info(
        dataset_name,
        template_data,
        template_v_ini,
        template_pivot_tokens,
        template_embeddings,
        template_to_param_id,
        template_to_event_id
    )
    
    return updated_model, updated_tokenizer, template_to_param_id

def replace_params_with_template_specific_tokens(dataset, template_to_param_id, split_regex):
    """将数据集中的参数占位符替换为模板特定的参数token"""
    updated_dataset = []
    
    for log, template, line_id, event_id in dataset:
        # 获取该日志对应的模板ID
        template_id = event_id
        
        # 如果该模板有对应的参数token
        if template_id in template_to_param_id:
            param_token_id = template_to_param_id[template_id]
            
            # 替换模板中的<*>为模板特定的PARAM token
            tokens_in_template = re.split(split_regex, template)
            for i, token in enumerate(tokens_in_template):
                if token == "<*>":
                    tokens_in_template[i] = f"PARAM{template_id}"
            
            # 重建模板
            updated_template = "".join(tokens_in_template)
            updated_dataset.append((log, updated_template, line_id, event_id))
        else:
            # 如果没有对应的参数token，保持原样
            updated_dataset.append((log, template, line_id, event_id))
    
    return updated_dataset 

