import os
import pandas as pd
import re

def save_results_to_csv(results, output_path):
    """
    将评估结果保存到CSV文件
    
    Args:
        results: 评估结果列表 [(original_log, predicted_template, true_template), ...]
        output_path: 输出文件路径
    """
    df = pd.DataFrame(results, columns=['OriginalLog', 'PredictedTemplate', 'TrueTemplate'])
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

# def merge_consecutive_wildcards(template):
#     """合并模板中连续的<*>标记"""
#     tokens = template.split()
#     merged_tokens = []
#     prev_token = None
    
#     for token in tokens:
#         if token == "<*>" and prev_token == "<*>":
#             continue
#         merged_tokens.append(token)
#         prev_token = token
    
#     return " ".join(merged_tokens)

def create_experiment_dir(dataset_name, base_dir='./experiments'):
    """创建实验目录"""
    # 获取当前日期和时间作为目录名的一部分
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建实验目录
    experiment_dir = os.path.join(base_dir, f"{dataset_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 创建子目录
    models_dir = os.path.join(experiment_dir, "models")
    results_dir = os.path.join(experiment_dir, "results")
    logs_dir = os.path.join(experiment_dir, "logs")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return {
        "base": experiment_dir,
        "models": models_dir,
        "results": results_dir,
        "logs": logs_dir
    }

def setup_logger(log_file):
    """设置日志记录器"""
    import logging
    
    # 创建日志记录器
    logger = logging.getLogger('log_parsing')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger