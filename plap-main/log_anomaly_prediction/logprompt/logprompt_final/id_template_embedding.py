from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
import re

# 加载BERT模型和tokenizer
model_name = '/home/zhouzt/experiment/log_parsing/roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()  # 设置为评估模式

# 检查CUDA可用性并将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型移动到GPU


# 定义预处理函数
def preprocess_text(text):
    # 只保留字母和数字字符
    return re.sub(r'[^a-zA-Z0-9]', ' ', text).strip()


def get_embedding(text, tokenizer, model, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 将输入移动到GPU
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # 获取[CLS]的embedding并移动到CPU


def extract_parameters(param_str):
    # 提取参数字符串中的值
    return eval(param_str)  # 使用eval将字符串转换为列表


def get_combined_embedding(param_list, tokenizer, model):
    embeddings = [get_embedding(param, tokenizer, model) for param in param_list]
    # 融合成一个768维的向量
    return torch.mean(torch.stack(embeddings), dim=0).numpy()


# 读取文件
file_path = '/home/zhouzt/experiment/log_anomaly_prediction/logparser/Drain/demo_result/Thunderbird_full.log_templates.csv'
df = pd.read_csv(file_path)

# 提取日志ID、模板和参数列表
log_ids = df['EventId'].tolist()
templates = [preprocess_text(template) for template in df['EventTemplate'].tolist()]
# parameters = [extract_parameters(param) for param in df['ParameterList'].tolist()]  # 移除参数信息

# 计算嵌入向量
log_id_embeddings = [get_embedding(log_id, tokenizer, model) for log_id in log_ids]
template_embeddings = [get_embedding(template, tokenizer, model) for template in templates]
# parameter_embeddings = [get_combined_embedding(param, tokenizer, model) for param in parameters]  # 移除参数嵌入计算

# 创建DataFrame以保存ID、嵌入后的ID、模板和参数列表嵌入向量
combined_df = pd.DataFrame({
    'EventId': log_ids,
    'EventIdEmbedding': list(log_id_embeddings),
    'EventTemplateEmbedding': list(template_embeddings),
    # 'ParameterListEmbedding': list(parameter_embeddings)  # 移除参数嵌入列
})

# 指定路径保存嵌入向量和ID
combined_embeddings_path = '/home/zhouzt/experiment/log_anomaly_prediction/logprompt/logprompt_final/thunderbird_id_template_embeddings.csv'

# 保存DataFrame为CSV文件
combined_df.to_csv(combined_embeddings_path, index=False)
