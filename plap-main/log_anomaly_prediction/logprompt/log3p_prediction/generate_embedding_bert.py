"""
使用bert-base-uncased生成EventId和EventTemplate的嵌入向量
生成新的嵌入文件，不会覆盖原有的嵌入文件（原有文件使用roberta-base生成）
"""
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import re
import os

# 配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "/home/zhouzt/experiment/log_parsing/bert-base-uncased"

# 数据集配置
DATASETS = {
    'BGL': {
        'template_file': '/home/zhouzt/experiment/log_anomaly_prediction/logparser/Drain/demo_result/BGL.log_templates.csv',
        'base_output_dir': '/home/zhouzt/experiment/log_anomaly_prediction/logprompt/BGL-120',
        'base_filename': 'bgl_id_template_embeddings'
    },
    'HDFS': {
        'template_file': '/home/zhouzt/experiment/log_anomaly_prediction/logparser/Drain/demo_result/HDFS.log_templates.csv',
        'base_output_dir': '/home/zhouzt/experiment/log_anomaly_prediction/logprompt/HDFS-56',
        'base_filename': 'hdfs_id_template_embeddings'
    }
}

def preprocess_text(text):
    """预处理文本：只保留字母和数字字符"""
    return re.sub(r'[^a-zA-Z0-9]', ' ', text).strip()

def get_embedding(text, tokenizer, model, max_length=512):
    """
    使用BERT模型获取文本的嵌入向量
    
    Args:
        text: 输入文本
        tokenizer: BERT tokenizer
        model: BERT模型
        max_length: 最大序列长度
    
    Returns:
        numpy数组，768维的嵌入向量
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 使用最后一层隐藏状态的[CLS] token表示
        # BERT的[CLS] token在位置0
        embedding = outputs.hidden_states[-1][:, 0, :].squeeze().cpu().numpy()
    
    return embedding

def generate_embeddings_for_dataset(dataset_name):
    """
    为指定数据集生成嵌入向量
    
    Args:
        dataset_name: 数据集名称 ('BGL' 或 'HDFS')
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"不支持的数据集: {dataset_name}，可选: {list(DATASETS.keys())}")
    
    config = DATASETS[dataset_name]
    template_file = config['template_file']
    base_output_dir = config['base_output_dir']
    base_filename = config['base_filename']
    
    # 生成新的文件名，包含模型名称，避免覆盖原有文件
    output_filename = f"{base_filename}_{MODEL_NAME.replace('-', '_')}.csv"
    output_file = os.path.join(base_output_dir, output_filename)
    output_dir = base_output_dir
    
    print(f"\n{'='*60}")
    print(f"开始为 {dataset_name} 数据集生成嵌入向量")
    print(f"使用模型: {MODEL_NAME}")
    print(f"模板文件: {template_file}")
    print(f"输出文件: {output_file}")
    print(f"{'='*60}\n")
    
    # 检查模板文件是否存在
    if not os.path.exists(template_file):
        raise FileNotFoundError(f"模板文件不存在: {template_file}")
    
    # 加载BERT模型和tokenizer
    print("正在加载BERT模型...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertModel.from_pretrained(MODEL_PATH)
    model.eval()
    model.to(DEVICE)
    print(f"模型已加载到 {DEVICE}\n")
    
    # 读取模板文件
    print("正在读取模板文件...")
    df = pd.read_csv(template_file)
    print(f"共读取 {len(df)} 条记录\n")
    
    # 提取EventId和EventTemplate
    log_ids = df['EventId'].tolist()
    templates = [preprocess_text(template) for template in df['EventTemplate'].tolist()]
    
    # 生成嵌入向量
    print("正在生成EventId嵌入向量...")
    log_id_embeddings = []
    for i, log_id in enumerate(log_ids):
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(log_ids)} 个EventId")
        embedding = get_embedding(log_id, tokenizer, model)
        log_id_embeddings.append(embedding)
    print(f"EventId嵌入向量生成完成，共 {len(log_id_embeddings)} 个\n")
    
    print("正在生成EventTemplate嵌入向量...")
    template_embeddings = []
    for i, template in enumerate(templates):
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(templates)} 个EventTemplate")
        embedding = get_embedding(template, tokenizer, model)
        template_embeddings.append(embedding)
    print(f"EventTemplate嵌入向量生成完成，共 {len(template_embeddings)} 个\n")
    
    # 创建DataFrame
    print("正在保存嵌入向量...")
    # 将嵌入向量转换为空格分隔的字符串格式，与原始格式保持一致
    # 这样parse_embedding函数才能正确解析
    combined_df = pd.DataFrame({
        'EventId': log_ids,
        'EventIdEmbedding': [' '.join(map(str, emb.tolist())) for emb in log_id_embeddings],
        'EventTemplateEmbedding': [' '.join(map(str, emb.tolist())) for emb in template_embeddings]
    })
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为CSV文件
    combined_df.to_csv(output_file, index=False)
    print(f"嵌入向量已保存到: {output_file}")
    print(f"文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    print(f"\n注意: 新文件已保存，原有嵌入文件未被覆盖")
    print(f"如需使用新嵌入文件，请修改config.py中的EMBEDDING_FILE_PATH")
    print(f"\n{'='*60}")
    print(f"{dataset_name} 数据集嵌入向量生成完成！")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import sys
    
    # 可以通过命令行参数指定数据集，或同时生成两个数据集
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1].upper()
        if dataset_name == "ALL":
            # 生成所有数据集的嵌入
            for ds_name in DATASETS.keys():
                generate_embeddings_for_dataset(ds_name)
        else:
            generate_embeddings_for_dataset(dataset_name)
    else:
        # 默认生成HDFS和BGL两个数据集
        print("未指定数据集，将生成所有数据集的嵌入向量")
        print("使用方法: python generate_embedding_bert.py [BGL|HDFS|ALL]")
        print("\n开始生成所有数据集的嵌入向量...\n")
        for dataset_name in DATASETS.keys():
            generate_embeddings_for_dataset(dataset_name)

