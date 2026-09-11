import re

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from collections import Counter
from avg_mac import Dtrain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练语言模型和分词器
model_name = "/home/zhouzt/experiment/log_parsing/roberta-base"  # 或其他预训练模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model=model.to(device)


# 动态扩充词汇表并调整模型嵌入
def add_new_tokens(tokens, tokenizer, model):
    new_tokens = [token for token in tokens if token not in tokenizer.get_vocab()]

    if new_tokens:
        # 添加新 tokens 到词汇表
        tokenizer.add_tokens(new_tokens)
        # 调整模型的嵌入层大小
        model.resize_token_embeddings(len(tokenizer))
        # 同步 vocab_size
        model.config.vocab_size = len(tokenizer)

    return new_tokens


# 利用预训练语言模型获取概率分布
def get_token_probabilities(model, tokenizer, X, device):
    # 假设 X 是原始的文本字符串，例如："Verification succeeded for <*>"
    tokens = re.split(r'[ ,\[\]]+', X)  # 按空格及分词符分割文本成 tokens 列表

    # 动态扩充词汇表
    add_new_tokens(tokens, tokenizer, model)

    # 将分割后的 tokens 转换为 token ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 创建 attention_mask，1 表示有效 token
    attention_mask = torch.ones(1, len(input_ids))  # 所有 token 都是有效的

    # 将输入转换为 tensor，并移动到 GPU（如果可用）
    inputs = {'input_ids': torch.tensor([input_ids]).to(device),
              'attention_mask': attention_mask.to(device)}

    # 使用模型进行推理
    with torch.no_grad():
        outputs = model(**inputs)  # 使用解包方式传递字典给模型

    # 获取 logits 和计算概率分布
    logits = outputs.logits  # 获取 logit 输出
    probabilities = torch.softmax(logits, dim=-1)  # 计算 softmax 概率分布
    # print("Probabilities shape:", probabilities.shape)

    return probabilities, inputs


# 提取参数位置并获取 Top-k 预测 Token
def extract_param_tokens(X, Y, tokenizer, probabilities, inputs, device, top_k=5):
    tokens_in_Y = re.split(r'[ ,\[\]]+', Y)  # 通过空格分割 Y
    param_positions = [i for i, token in enumerate(tokens_in_Y) if token == "<*>"]

    # 如果 Y 中没有 <*>，可以提前返回
    if not param_positions:
        # print("No <*> token found in Y.")
        return {}

    # 打印 token 的总长度
    # print("Total tokens in Y:", len(tokens_in_Y))

    top_k_tokens = {}
    for pos in param_positions:
        token_probs = probabilities[0, pos].to(device) # 获取该位置的概率
        top_k_indices = torch.topk(token_probs, top_k).indices
        top_k_tokens[pos] = [tokenizer.decode([idx]).strip() for idx in top_k_indices]

    return top_k_tokens


# 生成初始参数指示集合 V_ini
def generate_v_ini(top_k_tokens_list):
    v_ini = []
    for top_k_tokens in top_k_tokens_list:
        for tokens in top_k_tokens.values():
            v_ini.extend(tokens)
    return Counter(v_ini)


# 选择频率最高的 Token
def select_pivot_tokens(v_ini, top_n=8):
    tokens_to_remove = ["</s>", ".", "<s>", "]", "—", "[", ",", "...", ":", '"', ").", '-', '?', '(', '', '<unk>', '>',
                        '<mask>', ')', '=', '+', '_', '/']
    # 初始化过滤后的 tokens 列表
    filtered_tokens = []

    # 遍历最常见的 token，排除不需要的 token
    for token, freq in v_ini.most_common():
        if token not in tokens_to_remove:
            filtered_tokens.append(token)
        # 如果已经达到 top_n 个，退出循环
        if len(filtered_tokens) == top_n:
            break

    print("most_frequent_tokens:", filtered_tokens)
    return filtered_tokens


# 计算虚拟标签 Token “PARAM”的嵌入向量
def get_virtual_label_embedding(model, tokenizer, pivot_tokens, device):
    embeddings = []
    for token in pivot_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        embedding = model.roberta.embeddings.word_embeddings.weight[token_id].to(device)
        embeddings.append(embedding)
    virtual_embedding = torch.mean(torch.stack(embeddings), dim=0)
    print(virtual_embedding.shape)
    return virtual_embedding


# 将 PARAM 嵌入添加到模型中
def add_virtual_label_embedding(model, tokenizer, virtual_embedding, device):
    # 将虚拟标签嵌入到 GPU 上
    virtual_embedding = virtual_embedding.to(device)

    # 添加 PARAM 嵌入
    new_weight = torch.cat([model.roberta.embeddings.word_embeddings.weight, virtual_embedding.unsqueeze(0)])

    # 更新模型的词嵌入层
    model.roberta.embeddings.word_embeddings.weight = torch.nn.Parameter(new_weight)

    # 向分词器添加新的 token
    tokenizer.add_tokens(["PARAM"])

    model.resize_token_embeddings(len(tokenizer))

    # 同步模型的 vocab_size
    model.config.vocab_size = len(tokenizer)

    # 返回更新后的模型和分词器
    return model, tokenizer


# 处理所有日志并计算频率最高的 top 8 tokens
top_k_tokens_list = []

for log, template, line_id, event_id in Dtrain:
    probabilities, inputs = get_token_probabilities(model, tokenizer, log, device)
    top_k_tokens = extract_param_tokens(log, template, tokenizer, probabilities, inputs, device)
    top_k_tokens_list.append(top_k_tokens)

# 生成包含所有日志中 top-k tokens 的频率计数
v_ini = generate_v_ini(top_k_tokens_list)

# 选择频率最高的 8 个 tokens 作为虚拟标签
pivot_tokens = select_pivot_tokens(v_ini)

# 计算虚拟标签的嵌入向量
virtual_embedding = get_virtual_label_embedding(model, tokenizer, pivot_tokens, device)

# 添加虚拟标签嵌入
updated_model, updated_tokenizer = add_virtual_label_embedding(model, tokenizer, virtual_embedding, device)
