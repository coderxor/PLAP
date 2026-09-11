import argparse
import pandas as pd
import torch
import os
import numpy as np
import random
from log_parsing.log3p_parsing import config
from data_processor import sample_logs_by_event, create_data_loaders
from virtual_embedding import create_virtual_embedding
from train import train_model
from evaluate import eval_model, print_evaluation_results
from utils import save_results_to_csv, create_experiment_dir, setup_logger

def set_seed(seed):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保CuDNN是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def main():
    # 设置随机种子
    set_seed(config.RANDOM_SEED)
    
    parser = argparse.ArgumentParser(description='Log Parsing with Virtual Embedding')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pretrained model (default: use config)')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help=f'Number of training epochs (default: {config.EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help=f'Batch size for training (default: {config.BATCH_SIZE})')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='Directory to save models (default: ./models)')
    
    args = parser.parse_args()
    
    # 直接使用config中定义的数据集
    dataset_name = config.CURRENT_DATASET
    print(f"\n使用数据集: {dataset_name}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 获取数据集配置
    dataset_config = config.DATASETS[dataset_name]
    file_path = dataset_config['file_path']
    split_regex = dataset_config['split_regex']
    sample_count = dataset_config['sample_count']
    
    # 准备训练数据
    print(f"Preparing training data from {file_path}...")
    train_data = sample_logs_by_event(file_path, split_regex, n=sample_count)
    
    # 打印训练数据集大小
    print(f"训练数据集大小: {len(train_data)} 条日志")
    
    # 统计每个事件类型的日志数量
    event_counts = {}
    for _, _, _, event_id in train_data:
        if event_id not in event_counts:
            event_counts[event_id] = 0
        event_counts[event_id] += 1
    
    print(f"事件类型数量: {len(event_counts)}")
    # print("每种事件类型的日志数量:")
    # for event_id, count in sorted(event_counts.items()):
    #     print(f"  {event_id}: {count} 条日志")
    
    # 读取测试数据
    print("\nPreparing test data...")
    test_df = pd.read_csv(file_path)
    print(f"从CSV文件读取的总数据量: {len(test_df)} 条日志")
    
    test_data = []
    # 如果数据量少于100000，使用全部数据；否则采样100000条
    sample_size = min(100000, len(test_df))
    if sample_size < len(test_df):
        print(f"数据量大于100000，采样 {sample_size} 条作为测试数据")
        sampled_df = test_df.sample(n=sample_size, random_state=42)
    else:
        print(f"使用全部 {len(test_df)} 条数据作为测试数据")
        sampled_df = test_df
    
    for _, row in sampled_df.iterrows():
        try:
            line_id = row['LineId']
            log = row['Content']
            event_template = row['EventTemplate']
            event_id = row['EventId']
            
            # 检查必要字段是否为空或NaN
            if pd.isna(log) or pd.isna(event_template) or pd.isna(event_id):
                continue
            
            # 确保是字符串类型，并处理各种边界情况
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
            
            if event_id is None:
                continue
            else:
                event_id = str(event_id)
                if event_id.lower() == "nan" or event_id == "None":
                    continue
            
            test_data.append((log, event_template, line_id, event_id))
        except Exception as e:
            # 跳过有问题的行
            continue
    
    print(f"测试数据集大小: {len(test_data)} 条日志")
    
    # 创建虚拟嵌入
    print("\nCreating virtual embedding...")
    model_path = args.model_path or dataset_config.get('model_path', config.MODEL_NAME)
    model, tokenizer = create_virtual_embedding(train_data, split_regex, model_path)
    
    # 创建数据加载器
    print("Creating data loaders...")
    train_loader, test_loader = create_data_loaders(
        train_data, test_data, tokenizer, args.batch_size, config.DEVICE, split_regex
    )
    
    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # 创建实验目录
    experiment_dirs = create_experiment_dir(dataset_name)
    logger = setup_logger(os.path.join(experiment_dirs["logs"], "training.log"))
    
    # 记录数据集信息到日志
    logger.info(f"数据集: {dataset_name}")
    logger.info(f"训练数据集大小: {len(train_data)} 条日志")
    logger.info(f"事件类型数量: {len(event_counts)}")
    logger.info(f"测试数据集大小: {len(test_data)} 条日志")
    
    # 训练模型
    logger.info(f"Training model for {args.epochs} epochs...")
    best_model_path = train_model(
        model, train_loader, args.epochs, optimizer, tokenizer, 
        save_dir=experiment_dirs["models"], dataset_name=dataset_name
    )
    
    # 加载最佳模型
    logger.info(f"Loading best model from {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    # 评估模型
    logger.info("Evaluating model...")
    results, avg_loss, pa, ga = eval_model(model, test_loader, tokenizer)
    
    # 打印结果
    print_evaluation_results(results, avg_loss, pa, ga)
    
    # 保存结果
    results_file = os.path.join(experiment_dirs["results"], f"{dataset_name}_results.csv")
    save_results_to_csv(results, results_file)

if __name__ == "__main__":
    main()