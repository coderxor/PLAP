import re
import torch
from collections import Counter
from transformers import AutoModelForMaskedLM, AutoTokenizer
from log_parsing.log3p_parsing import config
from log_parsing.log3p_parsing.config import DEVICE, TOKENS_TO_REMOVE, TOP_K

def add_new_tokens(tokens, tokenizer, model):
    """动态扩充词汇表并调整模型嵌入"""
    new_tokens = [token for token in tokens if token not in tokenizer.get_vocab()]

    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)

    return new_tokens

def get_token_probabilities(model, tokenizer, text, split_regex, device):
    """利用预训练语言模型获取概率分布"""
    tokens = re.split(split_regex, text)
    
    add_new_tokens(tokens, tokenizer, model)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = torch.ones(1, len(input_ids))
    
    inputs = {
        'input_ids': torch.tensor([input_ids]).to(device),
        'attention_mask': attention_mask.to(device)
    }
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    
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

def generate_v_ini(top_k_tokens_list):
    """生成初始参数指示集合V_ini"""
    v_ini = []
    for top_k_tokens in top_k_tokens_list:
        for tokens in top_k_tokens.values():
            v_ini.extend(tokens)
    return Counter(v_ini)

def select_pivot_tokens(v_ini, tokens_to_remove=TOKENS_TO_REMOVE, top_n=None):
    """选择频率最高的Token"""
    # 如果没有指定top_n，则使用配置中的值（动态获取，支持实验时修改）
    if top_n is None:
        top_n = config.TOP_N
        
    filtered_tokens = []
    
    for token, freq in v_ini.most_common():
        # if token not in tokens_to_remove:
        filtered_tokens.append(token)
        if len(filtered_tokens) == top_n:
            break
    
    return filtered_tokens

def get_virtual_label_embedding(model, tokenizer, pivot_tokens, device):
    """计算虚拟标签Token "PARAM"的嵌入向量"""
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
    
    virtual_embedding = torch.mean(torch.stack(embeddings), dim=0)
    return virtual_embedding

def add_virtual_label_embedding(model, tokenizer, virtual_embedding, device):
    """将PARAM嵌入添加到模型中"""
    virtual_embedding = virtual_embedding.to(device)
    
    # 根据模型类型更新词嵌入
    if hasattr(model, 'roberta'):
        new_weight = torch.cat([model.roberta.embeddings.word_embeddings.weight, virtual_embedding.unsqueeze(0)])
        model.roberta.embeddings.word_embeddings.weight = torch.nn.Parameter(new_weight)
    elif hasattr(model, 'bert'):
        new_weight = torch.cat([model.bert.embeddings.word_embeddings.weight, virtual_embedding.unsqueeze(0)])
        model.bert.embeddings.word_embeddings.weight = torch.nn.Parameter(new_weight)
    elif hasattr(model, 'albert'):
        new_weight = torch.cat([model.albert.embeddings.word_embeddings.weight, virtual_embedding.unsqueeze(0)])
        model.albert.embeddings.word_embeddings.weight = torch.nn.Parameter(new_weight)
    else:
        raise ValueError("不支持的模型类型")
    
    tokenizer.add_tokens(["PARAM"])
    model.resize_token_embeddings(len(tokenizer))
    model.config.vocab_size = len(tokenizer)
    
    return model, tokenizer

def create_virtual_embedding(train_data, split_regex, model_name):
    """创建虚拟嵌入的主函数"""
    # 加载预训练模型 - 移除错误的参数
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32  # 保留这个参数是可以的
        # 移除 weights_only 参数，它不属于这个函数
    )
    model = model.to(DEVICE)
    
    # 处理所有日志并计算频率最高的tokens
    top_k_tokens_list = []
    
    for log, template, line_id, event_id in train_data:
        probabilities, inputs = get_token_probabilities(model, tokenizer, log, split_regex, DEVICE)
        top_k_tokens = extract_param_tokens(log, template, tokenizer, probabilities, inputs, split_regex, DEVICE)
        top_k_tokens_list.append(top_k_tokens)
    
    # 生成包含所有日志中top-k tokens的频率计数
    v_ini = generate_v_ini(top_k_tokens_list)
    
    # 选择频率最高的tokens作为虚拟标签
    pivot_tokens = select_pivot_tokens(v_ini)
    
    # 计算虚拟标签的嵌入向量
    virtual_embedding = get_virtual_label_embedding(model, tokenizer, pivot_tokens, DEVICE)
    
    # 添加虚拟标签嵌入
    updated_model, updated_tokenizer = add_virtual_label_embedding(model, tokenizer, virtual_embedding, DEVICE)
    
    return updated_model, updated_tokenizer