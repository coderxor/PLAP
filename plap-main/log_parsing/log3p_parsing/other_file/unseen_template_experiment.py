import os
import pandas as pd
import numpy as np
import torch
import random
import time
from tqdm import tqdm
import re

from log_parsing.log3p_parsing import config
from data_processor import create_data_loaders
from virtual_embedding import create_virtual_embedding
from train import train_model
from evaluate import eval_model, print_evaluation_results
from utils import save_results_to_csv, create_experiment_dir, setup_logger

# ======== 实验配置 ========
# 修改这里的变量来配置实验
DATASET = 'android'  # 选择数据集: 'hdfs', 'bgl', 'android', 'mac', 'zookeeper'
TRAIN_TEMPLATE_COUNT = 133  # 用于训练的模板数量：80%
SAMPLES_PER_TEMPLATE = 4  # 每个模板抽取的样本数
EPOCHS = 100               # 训练轮数
RANDOM_SEED = 42           # 全局随机种子
# ========================

# 全局随机种子设置函数
def set_random_seed(seed):
    """设置所有随机数生成器的种子，确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def sample_logs_by_template_count(file_path, split_regex, train_template_count, samples_per_template=3, random_state=42):
    """
    将日志按指定的训练模板数量分割成训练集和测试集
    
    Args:
        file_path: 结构化日志文件路径
        split_regex: 分割日志内容的正则表达式
        train_template_count: 用于训练的模板数量
        samples_per_template: 每个模板抽取的样本数
        random_state: 随机种子
    
    Returns:
        train_data: 训练数据列表 [(log, template, line_id, event_id), ...]
        test_data: 测试数据列表 [(log, template, line_id, event_id), ...]
        train_templates: 训练集中的模板列表
        test_templates: 测试集中的模板列表
    """
    # 设置随机种子
    random.seed(random_state)
    np.random.seed(random_state)
    
    # 读取结构化日志文件
    structured_logs_df = pd.read_csv(file_path)
    
    # 获取所有唯一的模板和对应的EventId
    templates_df = structured_logs_df[['EventTemplate', 'EventId']].drop_duplicates()
    templates = templates_df['EventTemplate'].values
    total_templates = len(templates)
    
    # 确保训练模板数量不超过总模板数
    train_template_count = min(train_template_count, total_templates)
    
    # 随机选择训练模板
    shuffled_indices = np.random.permutation(total_templates)
    train_indices = shuffled_indices[:train_template_count]
    test_indices = shuffled_indices[train_template_count:]
    
    train_templates = templates[train_indices]
    test_templates = templates[test_indices]
    
    print(f"总模板数: {total_templates}")
    print(f"训练集模板数: {len(train_templates)}")
    print(f"测试集模板数: {len(test_templates)}")
    
    # 准备训练数据和测试数据
    train_data = []
    test_data = []
    
    # 处理训练集模板
    for template in train_templates:
        template_logs = structured_logs_df[structured_logs_df['EventTemplate'] == template]
        
        # 抽样
        if len(template_logs) < samples_per_template:
            sampled_logs = template_logs.sample(n=len(template_logs), random_state=random_state)
            # 如果样本不足，通过重复采样来补充
            while len(sampled_logs) < samples_per_template:
                additional = template_logs.sample(n=1, random_state=random_state)
                sampled_logs = pd.concat([sampled_logs, additional])
        else:
            sampled_logs = template_logs.sample(n=samples_per_template, random_state=random_state)
        
        # 收集日志样本
        for _, row in sampled_logs.iterrows():
            line_id = row['LineId']
            log = row['Content']
            event_template = row['EventTemplate']
            event_id = row['EventId']
            train_data.append((log, event_template, line_id, event_id))
    
    # 处理测试集模板
    for template in test_templates:
        template_logs = structured_logs_df[structured_logs_df['EventTemplate'] == template]
        
        # 抽样
        if len(template_logs) < samples_per_template:
            sampled_logs = template_logs.sample(n=len(template_logs), random_state=random_state)
            # 如果样本不足，通过重复采样来补充
            while len(sampled_logs) < samples_per_template:
                additional = template_logs.sample(n=1, random_state=random_state)
                sampled_logs = pd.concat([sampled_logs, additional])
        else:
            sampled_logs = template_logs.sample(n=samples_per_template, random_state=random_state)
        
        # 收集日志样本
        for _, row in sampled_logs.iterrows():
            line_id = row['LineId']
            log = row['Content']
            event_template = row['EventTemplate']
            event_id = row['EventId']
            test_data.append((log, event_template, line_id, event_id))
    
    return train_data, test_data, train_templates, test_templates

def get_dataset_template_count(file_path):
    """获取数据集中的模板总数"""
    structured_logs_df = pd.read_csv(file_path)
    return len(structured_logs_df['EventTemplate'].unique())

def prepare_challenging_training_data(file_path, split_regex, ratio=0.3):
    """准备更具挑战性的训练数据，故意留出一些位置类型"""
    structured_logs_df = pd.read_csv(file_path)
    
    # 分析所有参数的位置类型
    position_types = []
    
    for _, row in structured_logs_df.iterrows():
        template = row['EventTemplate']
        template_tokens = re.split(split_regex, template)
        
        for i, token in enumerate(template_tokens):
            if token == "<*>":
                # 记录参数前后的上下文
                prev_token = template_tokens[i-1] if i > 0 else "START"
                next_token = template_tokens[i+1] if i < len(template_tokens) - 1 else "END"
                position_type = f"{prev_token}_{next_token}"
                position_types.append(position_type)
    
    # 获取唯一的位置类型
    unique_position_types = list(set(position_types))
    
    # 随机留出一部分位置类型
    np.random.shuffle(unique_position_types)
    holdout_count = int(len(unique_position_types) * ratio)
    holdout_types = set(unique_position_types[:holdout_count])
    
    # 筛选训练数据
    train_data = []
    holdout_data = []
    
    for _, row in structured_logs_df.iterrows():
        log = row['Content']
        template = row['EventTemplate']
        line_id = row['LineId']
        event_id = row['EventId']
        
        template_tokens = re.split(split_regex, template)
        has_holdout_type = False
        
        for i, token in enumerate(template_tokens):
            if token == "<*>":
                prev_token = template_tokens[i-1] if i > 0 else "START"
                next_token = template_tokens[i+1] if i < len(template_tokens) - 1 else "END"
                position_type = f"{prev_token}_{next_token}"
                
                if position_type in holdout_types:
                    has_holdout_type = True
                    break
        
        if has_holdout_type:
            holdout_data.append((log, template, line_id, event_id))
        else:
            train_data.append((log, template, line_id, event_id))
    
    return train_data, holdout_data

def run_unseen_template_experiment():
    """
    运行未知模板解析实验
    """
    # 设置全局随机种子
    set_random_seed(RANDOM_SEED)
    
    # 设置数据集
    config.CURRENT_DATASET = DATASET
    print(f"\n使用数据集: {DATASET}")
    
    # 创建时间戳，用于创建唯一的实验目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir_name = f"{DATASET}_unseen_template_{TRAIN_TEMPLATE_COUNT}_{timestamp}"
    
    # 创建实验目录
    experiment_dirs = create_experiment_dir(experiment_dir_name)
    logger = setup_logger(os.path.join(experiment_dirs["logs"], "experiment.log"))
    
    # 获取数据集配置
    dataset_config = config.get_current_dataset_config()
    file_path = dataset_config['file_path']
    split_regex = dataset_config['split_regex']
    
    # 获取数据集的模板总数
    total_template_count = get_dataset_template_count(file_path)
    # 使用普通的模板划分方法
    logger.info(f"准备数据集: {DATASET}，训练模板数量: {TRAIN_TEMPLATE_COUNT}，每模板样本数: {SAMPLES_PER_TEMPLATE}")
    actual_train_count = min(TRAIN_TEMPLATE_COUNT, total_template_count)
    train_data, test_data, train_templates, test_templates = sample_logs_by_template_count(
        file_path,
        split_regex,
        train_template_count=actual_train_count,
        samples_per_template=SAMPLES_PER_TEMPLATE
    )

    # 记录训练和测试模板信息
    logger.info(f"训练集模板数: {len(train_templates)}")
    logger.info(f"测试集模板数: {len(test_templates)}")
    logger.info(f"训练集样本数: {len(train_data)}")
    logger.info(f"测试集样本数: {len(test_data)}")
    
    # 保存训练和测试模板以便后续分析
    pd.DataFrame({'TrainTemplates': train_templates}).to_csv(
        os.path.join(experiment_dirs["results"], "train_templates.csv"), index=False
    )
    pd.DataFrame({'TestTemplates': test_templates}).to_csv(
        os.path.join(experiment_dirs["results"], "test_templates.csv"), index=False
    )
    
    # 创建虚拟嵌入
    logger.info("创建虚拟嵌入...")
    # 只使用训练数据创建词汇表
    model, tokenizer = create_virtual_embedding(train_data, split_regex, config.MODEL_NAME)
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, test_loader = create_data_loaders(
        train_data, test_data, tokenizer, config.BATCH_SIZE, config.DEVICE, split_regex
    )
    
    # 设置优化器，降低学习率
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # 训练模型
    logger.info(f"训练模型，迭代次数: {EPOCHS}...")
    best_model_path = train_model(
        model, train_loader, EPOCHS, optimizer, tokenizer,
        save_dir=experiment_dirs["models"], dataset_name=f"{DATASET}_unseen"
    )
    
    # 加载最佳模型
    logger.info(f"加载最佳模型: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    # 评估模型
    logger.info("对未知模板进行评估...")
    results, avg_loss, pa, ga = eval_model(model, test_loader, tokenizer)
    
    # 记录评估结果
    logger.info(f"未知模板解析结果 - PA: {pa:.4f}, GA: {ga:.4f}")
    
    # 保存结果
    results_file = os.path.join(experiment_dirs["results"], f"{DATASET}_unseen_results.csv")
    save_results_to_csv(results, results_file)
    
    # 保存评估指标
    metrics_df = pd.DataFrame({
        'Dataset': [DATASET],
        'TrainTemplateCount': [actual_train_count],
        'TotalTemplateCount': [total_template_count],
        'SamplesPerTemplate': [SAMPLES_PER_TEMPLATE],
        'TrainTemplates': [len(train_templates)],
        'TestTemplates': [len(test_templates)],
        'PA': [pa],
        'GA': [ga]
    })
    metrics_file = os.path.join(experiment_dirs["results"], "metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    
    print("\n===== 未知模板实验结果 =====")
    print(f"数据集: {DATASET}")
    print(f"训练模板数量: {len(train_templates)}")
    print(f"测试模板数量: {len(test_templates)}")
    print(f"解析准确率 (PA): {pa:.2%}")
    print(f"组准确度 (GA): {ga:.2%}")
    
    return {
        'dataset': DATASET,
        'train_template_count': actual_train_count,
        'total_template_count': total_template_count,
        'test_template_count': len(test_templates),
        'pa': pa,
        'ga': ga,
        'results_dir': experiment_dirs["results"]
    }

if __name__ == "__main__":
    # 显示当前配置
    print("===== 实验配置 =====")
    print(f"数据集: {DATASET}")
    print(f"训练模板数量: {TRAIN_TEMPLATE_COUNT}")
    print(f"每模板样本数: {SAMPLES_PER_TEMPLATE}")
    print(f"训练轮数: {EPOCHS}")
    print("==================\n")
    
    # 运行实验
    result = run_unseen_template_experiment()
    
    # 显示结果保存位置
    print(f"\n实验结果已保存到: {result['results_dir']}") 