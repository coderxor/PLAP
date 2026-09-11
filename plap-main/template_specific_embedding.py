import re
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from collections import defaultdict

def get_token_probabilities(model, tokenizer, text, template, split_regex, device):
    """利用预训练语言模型获取概率分布"""
    tokens = re.split(split_regex, text)
    template_tokens = re.split(split_regex, template)
    
    # 找到参数位置
    param_positions = [i for i, token in enumerate(template_tokens) if token == "<*>"]
    
    add_new_tokens(tokens, tokenizer, model)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = torch.ones(1, len(input_ids))
    
    # 创建参数位置的掩码
    param_mask = torch.ones_like(attention_mask)
    for pos in param_positions:
        if pos < param_mask.size(1):
            param_mask[0, pos] = 0
    
    inputs = {
        'input_ids': torch.tensor([input_ids]).to(device),
        'attention_mask': attention_mask.to(device),
        'param_attention_mask': param_mask.to(device)  # 添加参数掩码
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