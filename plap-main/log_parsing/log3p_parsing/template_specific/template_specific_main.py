import argparse
import os
import torch
import random
import numpy as np
import pandas as pd
from log_parsing.log3p_parsing.template_specific.template_specific_config import *
from log_parsing.log3p_parsing.template_specific.template_specific_processor import sample_logs_by_event, create_template_specific_data_loaders
from log_parsing.log3p_parsing.template_specific.template_specific_embedding import create_template_specific_embeddings
from log_parsing.log3p_parsing.template_specific.template_specific_evaluate import eval_template_specific_model, print_template_specific_results
from log_parsing.log3p_parsing.train import train_model  # 可以复用原有的训练函数
from log_parsing.log3p_parsing.utils import save_results_to_csv, create_experiment_dir, setup_logger

def set_seed(seed):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 确保CuDNN是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    # 设置随机种子
    set_seed(RANDOM_SEED)
    
    parser = argparse.ArgumentParser(description='Log Parsing with Template-Specific Parameters')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model (default: use config)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--save_dir', type=str, default='./template_specific_models',
                        help='Directory to save models (default: ./template_specific_models)')
    
    args = parser.parse_args()
    
    # 获取数据集配置
    dataset_name = CURRENT_DATASET
    dataset_config = DATASETS[dataset_name]
    file_path = dataset_config['file_path']
    split_regex = dataset_config['split_regex']
    sample_count = dataset_config['sample_count']
    
    # 准备训练数据
    print(f"\n准备训练数据，从 {file_path} ...")
    train_data = sample_logs_by_event(file_path, split_regex, n=sample_count)
    print(f"训练数据集大小: {len(train_data)} 条日志")
    
    # 准备测试数据
    print("\n准备测试数据...")
    test_df = pd.read_csv(file_path)
    test_size = 1  # 使用20%的数据作为测试集
    test_df = test_df.sample(frac=test_size, random_state=42)


    # 直接使用全部数据，而不是随机抽样
    test_data = []
    for _, row in test_df.iterrows():
        line_id = row['LineId']
        log = row['Content']
        event_template = row['EventTemplate']
        event_id = row['EventId']
        # 如果event_id是字符串形式（如'E1'），转换为数字
        template_id = int(event_id[1:]) if event_id.startswith('E') else int(event_id)
        test_data.append((log, event_template, line_id, template_id))
    print(f"测试数据集大小: {len(test_data)} 条日志")
    
    # 创建模板特定的嵌入，传入数据集名称的大写形式
    print("\n创建模板特定的参数嵌入...")
    model, tokenizer, template_to_param_id = create_template_specific_embeddings(
        train_data, 
        split_regex, 
        args.model_path or MODEL_NAME,
        dataset_name=dataset_name.upper()  # 使用配置中的数据集名称，并转为大写
    )
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, test_loader = create_template_specific_data_loaders(
        train_data, 
        test_data,
        tokenizer,
        template_to_param_id,
        args.batch_size,
        DEVICE,
        split_regex
    )
    
    # 创建实验目录
    experiment_dirs = create_experiment_dir(f"{dataset_name}_template_specific")
    logger = setup_logger(os.path.join(experiment_dirs["logs"], "training.log"))
    
    # 训练模型
    print(f"\n开始训练模型，训练轮数: {args.epochs}")
    best_model_path = train_model(
        model, 
        train_loader,
        args.epochs,
        torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE),
        tokenizer,
        save_dir=experiment_dirs["models"],
        dataset_name=f"{dataset_name}_template_specific"
    )
    
    # 加载最佳模型并评估
    print(f"\n加载最佳模型进行评估: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    results, avg_loss, template_pa, ga, output_data = eval_template_specific_model(
        model, 
        test_loader,
        tokenizer,
        template_to_param_id
    )
    
    # 打印和保存结果
    print_template_specific_results(results, avg_loss, template_pa, ga)
    save_results_to_csv(
        output_data,
        os.path.join(experiment_dirs["results"], f"{dataset_name}_template_specific_results.csv")
    )

if __name__ == "__main__":
    main() 

