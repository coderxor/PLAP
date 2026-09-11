"""
滑动窗口大小对实验结果的影响实验
横坐标：窗口大小（WINDOW_DURATION和NEXT_WINDOW_DURATION保持一致）
纵坐标：Accuracy, Precision, Recall, F1 (0-1)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import logging
from tqdm import tqdm

# 导入必要的模块
from log_anomaly_prediction.logprompt.log3p_prediction.config import *
from log_anomaly_prediction.logprompt.log3p_prediction.data.dataset import create_data_loaders
from log_anomaly_prediction.logprompt.log3p_prediction.models.lstm_attention import LSTMWithAttention
from log_anomaly_prediction.logprompt.log3p_prediction.train import train_model
from log_anomaly_prediction.logprompt.log3p_prediction.evaluate import evaluate_model

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')  # 使用无头后端，适合服务器环境

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_dataset_config(config_module, dataset_name, window_size):
    """reload 之后按指定数据集重写路径和超参，避免一直用 config.py 里写死的 HDFS。"""
    config_module.DATASET = dataset_name
    config_module.WINDOW_DURATION = window_size
    config_module.NEXT_WINDOW_DURATION = window_size
    if dataset_name == 'BGL':
        cfg = config_module.BGL_CONFIG
        config_module.LOG_FILE_PATH = config_module.BGL_LOG_FILE_PATH
        config_module.EMBEDDING_FILE_PATH = config_module.BGL_EMBEDDING_FILE_PATH
        config_module.LABEL_FILE_PATH = None
    else:
        cfg = config_module.HDFS_CONFIG
        config_module.LOG_FILE_PATH = config_module.HDFS_LOG_FILE_PATH
        config_module.EMBEDDING_FILE_PATH = config_module.HDFS_EMBEDDING_FILE_PATH
        config_module.LABEL_FILE_PATH = config_module.HDFS_LABEL_FILE_PATH
    config_module.CONFIG = cfg
    for key, value in cfg.items():
        setattr(config_module, key, value)


def run_experiment_for_window_size(window_size, dataset_name=None):
    """
    对指定的窗口大小运行完整的训练和评估流程
    
    Args:
        window_size: 窗口大小（WINDOW_DURATION和NEXT_WINDOW_DURATION都设置为这个值）
        dataset_name: 数据集名称，None 则使用 config 当前值
    
    Returns:
        dict: 包含评估指标的字典
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"开始实验：数据集 = {dataset_name or DATASET}，窗口大小 = {window_size}")
    logger.info(f"{'='*60}\n")
    
    import importlib
    import log_anomaly_prediction.logprompt.log3p_prediction.config as config_module
    importlib.reload(config_module)
    if dataset_name is None:
        dataset_name = config_module.DATASET
    apply_dataset_config(config_module, dataset_name, window_size)
    
    import log_anomaly_prediction.logprompt.log3p_prediction.data.dataset as dataset_mod
    import log_anomaly_prediction.logprompt.log3p_prediction.data.data_loader as loader_mod
    import log_anomaly_prediction.logprompt.log3p_prediction.models.lstm_attention as model_mod
    import log_anomaly_prediction.logprompt.log3p_prediction.evaluate as eval_mod
    importlib.reload(dataset_mod)
    importlib.reload(loader_mod)
    importlib.reload(model_mod)
    importlib.reload(eval_mod)
    
    DATASET = config_module.DATASET
    DEVICE = config_module.DEVICE
    RANDOM_SEED = config_module.RANDOM_SEED
    SLIDE_INTERVAL = config_module.SLIDE_INTERVAL
    INTERVAL = config_module.INTERVAL
    N_ROWS = config_module.N_ROWS
    EMBEDDING_PARTS = config_module.EMBEDDING_PARTS
    EMBEDDING_DIM = config_module.EMBEDDING_DIM
    NUM_EPOCHS = config_module.NUM_EPOCHS
    LOG_FILE_PATH = config_module.LOG_FILE_PATH
    LABEL_FILE_PATH = config_module.LABEL_FILE_PATH
    LSTMWithAttention = model_mod.LSTMWithAttention
    create_data_loaders = dataset_mod.create_data_loaders
    evaluate_model = eval_mod.evaluate_model
    load_and_process_bgl_data = loader_mod.load_and_process_bgl_data
    load_and_process_hdfs_data = loader_mod.load_and_process_hdfs_data
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        logger.info(f"加载和处理 {DATASET} 数据（窗口大小={window_size}）...")
        
        if DATASET == 'BGL':
            texts, labels = load_and_process_bgl_data(
                LOG_FILE_PATH,
                window_duration=window_size,
                slide_interval=SLIDE_INTERVAL,
                next_window_duration=window_size,
                interval=INTERVAL,
                n_rows=N_ROWS
            )
        elif DATASET == 'HDFS':
            texts, labels = load_and_process_hdfs_data(
                LOG_FILE_PATH,
                label_file_path=LABEL_FILE_PATH,
                window_duration=window_size,
                slide_interval=SLIDE_INTERVAL,
                next_window_duration=window_size,
                interval=INTERVAL,
                n_rows=N_ROWS
            )
        else:
            raise ValueError(f"不支持的数据集: {DATASET}")
        
        if len(texts) == 0:
            logger.warning(f"窗口大小 {window_size} 没有生成任何数据，跳过")
            return None
        
        # 分割数据集
        logger.info("分割数据集...")
        X_train, X_test, y_train, y_test = loader_mod.split_dataset(texts, labels)
        
        # 创建数据加载器
        logger.info("创建数据加载器...")
        train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test)
        
        # 创建模型
        logger.info("创建模型...")
        input_size = EMBEDDING_PARTS * EMBEDDING_DIM * 2
        model = LSTMWithAttention(input_size=input_size).to(DEVICE)
        
        # 为每个窗口大小使用不同的模型保存路径，避免覆盖
        # 将窗口大小转换为字符串，浮点数用下划线代替小数点
        window_size_str = str(window_size).replace('.', '_')
        # 为这一组实验增加后缀 _ws4，避免与之前实验的模型文件重名
        if DATASET == 'BGL':
            model_save_path = f'best_model_bgl_window_{window_size_str}_ws4.pth'
        else:
            model_save_path = f'best_model_hdfs_window_{window_size_str}_ws4.pth'
        
        # 临时修改模型保存路径
        original_bgl_path = config_module.BGL_MODEL_SAVE_PATH
        original_hdfs_path = config_module.HDFS_MODEL_SAVE_PATH
        
        config_module.BGL_MODEL_SAVE_PATH = model_save_path
        config_module.HDFS_MODEL_SAVE_PATH = model_save_path
        
        # 重新导入train模块以确保使用最新的保存路径
        from importlib import reload
        import log_anomaly_prediction.logprompt.log3p_prediction.train as train_module
        reload(train_module)
        from log_anomaly_prediction.logprompt.log3p_prediction.train import train_model
        
        # 训练模型
        logger.info(f"训练模型（{NUM_EPOCHS} epochs）...")
        train_model(model, train_loader, NUM_EPOCHS)
        
        # 评估模型
        logger.info("评估模型...")
        metrics = evaluate_model(model, test_loader, model_path=model_save_path)
        
        # 恢复原始路径
        config_module.BGL_MODEL_SAVE_PATH = original_bgl_path
        config_module.HDFS_MODEL_SAVE_PATH = original_hdfs_path
        
        logger.info(f"窗口大小 {window_size} 的实验完成")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}\n")
        
        return metrics
    except Exception as e:
        logger.error(f"窗口大小 {window_size} 的实验失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def plot_results(results_df, dataset_name, output_dir='/home/zhouzt/experiment/log_anomaly_prediction/logprompt/log3p_prediction/window_size_experiment'):
    """
    绘制折线图
    
    Args:
        results_df: 包含实验结果的DataFrame
        dataset_name: 数据集名称
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制折线
    plt.plot(results_df['window_size'], results_df['accuracy'], marker='o', label='Accuracy', linewidth=2, markersize=8)
    plt.plot(results_df['window_size'], results_df['precision'], marker='s', label='Precision', linewidth=2, markersize=8)
    plt.plot(results_df['window_size'], results_df['recall'], marker='^', label='Recall', linewidth=2, markersize=8)
    plt.plot(results_df['window_size'], results_df['f1'], marker='d', label='F1', linewidth=2, markersize=8)
    
    # 设置标签和标题
    plt.xlabel('Window Size', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    plt.title(f'Impact of Window Size on Model Performance ({dataset_name})', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(0, 1)  # 确保y轴范围在0-1之间
    
    # 设置x轴刻度
    # 如果窗口大小是浮点数，格式化显示标签
    if results_df['window_size'].dtype == float:
        x_labels = [f'{x:.1f}' for x in results_df['window_size']]
    else:
        x_labels = [str(x) for x in results_df['window_size']]
    
    plt.xticks(results_df['window_size'], x_labels, fontsize=11, rotation=0)
    plt.yticks(fontsize=11)
    
    # 保存图片（本组实验使用后缀 _ws4，避免覆盖之前的图像）
    output_file = os.path.join(output_dir, f'window_size_experiment_{dataset_name}_ws4.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"折线图已保存到: {output_file}")
    
    # 也保存为PDF格式
    output_file_pdf = os.path.join(output_dir, f'window_size_experiment_{dataset_name}_ws4.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    
    plt.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Window Size Experiment')
    parser.add_argument('--window_sizes', type=str, default=None,
                       help='窗口大小列表，用逗号分隔。不指定则默认 1 到 10')
    parser.add_argument('--dataset', type=str, default=None, choices=['BGL', 'HDFS'],
                       help='数据集名称。不指定则依次跑 HDFS 和 BGL')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/zhouzt/experiment/log_anomaly_prediction/logprompt/log3p_prediction/window_size_experiment',
                       help='结果输出目录')
    args = parser.parse_args()
    
    if args.window_sizes is None:
        window_sizes = list(range(1, 11))
    else:
        window_sizes = [float(x.strip()) for x in args.window_sizes.split(',')]
        if all(float(x).is_integer() for x in window_sizes):
            window_sizes = [int(x) for x in window_sizes]
    
    window_sizes = sorted(window_sizes)
    datasets = [args.dataset] if args.dataset else ['HDFS', 'BGL']
    
    logger.info(f"开始窗口大小实验")
    logger.info(f"数据集: {datasets}")
    logger.info(f"窗口大小列表: {window_sizes}")
    logger.info(f"输出目录: {args.output_dir}\n")
    
    for dataset_name in datasets:
        logger.info(f"======== 数据集 {dataset_name} ========")
        results = []
        for window_size in tqdm(window_sizes, desc=f"{dataset_name} 实验进度"):
            metrics = run_experiment_for_window_size(window_size, dataset_name=dataset_name)
            
            if metrics is not None:
                results.append({
                    'window_size': window_size,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1']
                })
            else:
                logger.warning(f"{dataset_name} 窗口大小 {window_size} 的实验失败，跳过")
        
        if len(results) == 0:
            logger.error(f"{dataset_name} 没有成功的实验结果！")
            continue
        
        results_df = pd.DataFrame(results)
        os.makedirs(args.output_dir, exist_ok=True)
        csv_file = os.path.join(args.output_dir, f'window_size_experiment_{dataset_name}_ws4.csv')
        results_df.to_csv(csv_file, index=False)
        logger.info(f"\n{dataset_name} 实验结果已保存到: {csv_file}")
        
        print("\n" + "="*60)
        print(f"实验结果汇总 ({dataset_name})")
        print("="*60)
        print(results_df.to_string(index=False))
        print("="*60 + "\n")
        
        logger.info(f"绘制 {dataset_name} 折线图...")
        plot_results(results_df, dataset_name, args.output_dir)
    
    logger.info("实验完成！")


if __name__ == "__main__":
    main()

