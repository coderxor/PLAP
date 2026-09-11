import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from log_parsing.log3p_parsing import config
from log_parsing.log3p_parsing.data_processor import sample_logs_by_event, create_data_loaders
from log_parsing.log3p_parsing.virtual_embedding import create_virtual_embedding
from log_parsing.log3p_parsing.train import train_model
from log_parsing.log3p_parsing.evaluate import eval_model
from log_parsing.log3p_parsing.utils import create_experiment_dir, setup_logger

def set_seed(seed):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def run_top_n_experiment():
    """运行不同TOP_N值的对比实验"""
    # 设置随机种子以确保可重复性
    set_seed(config.RANDOM_SEED)
    
    # 创建实验目录
    experiment_dirs = create_experiment_dir("top_n_experiment")
    logger = setup_logger(os.path.join(experiment_dirs["logs"], "experiment.log"))
    
    # 存储实验结果
    results = {
        dataset: {
            "pa": [],  # 解析准确率
            "ga": [],  # 组准确度
            "pa_smooth": [],  # 平滑后的解析准确率
            "ga_smooth": []   # 平滑后的组准确度
        } for dataset in config.EXPERIMENT["datasets"]
    }
    
    for top_n in config.EXPERIMENT["top_n_range"]:
        # 为每个数据集分别创建结果收集器
        current_results = {
            dataset_name: {"pa": None, "ga": None} for dataset_name in config.EXPERIMENT["datasets"]
        }
        
        # 由于你不需要重复实验，把n_repeats设为1就可以了
        logger.info(f"Testing with TOP_N = {top_n}")
        
        # 临时修改配置中的TOP_N值
        config.TOP_N = top_n
        
        # 对每个数据集进行测试
        for dataset_name in config.EXPERIMENT["datasets"]:
            logger.info(f"Processing dataset: {dataset_name}")
            
            # 设置当前数据集
            config.CURRENT_DATASET = dataset_name
            
            # 获取数据集配置
            dataset_config = config.get_current_dataset_config()
            file_path = dataset_config['file_path']
            split_regex = dataset_config['split_regex']
            sample_count = dataset_config['sample_count']
            
            try:
                # 准备训练数据
                train_data = sample_logs_by_event(file_path, split_regex, n=sample_count)
                
                # 读取测试数据
                test_df = pd.read_csv(file_path)
                test_data = []
                for _, row in test_df.sample(n=2000, random_state=42).iterrows():
                    line_id = row['LineId']
                    log = row['Content']
                    event_template = row['EventTemplate']
                    event_id = row['EventId']
                    test_data.append((log, event_template, line_id, event_id))
                
                # 创建虚拟嵌入
                model, tokenizer = create_virtual_embedding(train_data, split_regex, config.MODEL_NAME)
                
                # 创建数据加载器
                train_loader, test_loader = create_data_loaders(
                    train_data, test_data, tokenizer, config.BATCH_SIZE, config.DEVICE, split_regex
                )
                
                # 设置优化器
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
                
                # 训练模型
                best_model_path = train_model(
                    model, train_loader, epochs=config.EPOCHS, optimizer=optimizer, tokenizer=tokenizer,
                    save_dir=experiment_dirs["models"], dataset_name=f"{dataset_name}_top_n_{top_n}"
                )
                
                # 加载最佳模型
                model.load_state_dict(torch.load(best_model_path, weights_only=True))
                
                # 评估模型
                _, avg_loss, pa, ga = eval_model(model, test_loader, tokenizer)
                
                # 直接存储当前数据集的结果
                current_results[dataset_name]["pa"] = pa
                current_results[dataset_name]["ga"] = ga
                
                logger.info(f"Dataset: {dataset_name}, TOP_N: {top_n}, PA: {pa:.4f}, GA: {ga:.4f}")
                
            except Exception as e:
                logger.error(f"Error processing {dataset_name} with TOP_N={top_n}: {str(e)}")
                # 记录失败的结果为0
                current_results[dataset_name]["pa"] = 0
                current_results[dataset_name]["ga"] = 0
        
        # 将当前TOP_N的结果添加到总结果中
        for dataset_name in config.EXPERIMENT["datasets"]:
            # 确保数据集在结果字典中存在
            if dataset_name not in results:
                results[dataset_name] = {"pa": [], "ga": [], "pa_smooth": [], "ga_smooth": []}
            
            # 添加当前top_n的结果到相应数据集的列表中
            pa_value = current_results[dataset_name]["pa"]
            ga_value = current_results[dataset_name]["ga"]
            
            # 确保值不是None
            results[dataset_name]["pa"].append(pa_value if pa_value is not None else 0)
            results[dataset_name]["ga"].append(ga_value if ga_value is not None else 0)
    
    # 2. 使用更强的平滑方法
    def smooth_results(values, window_size=5):
        """使用加权移动平均进行平滑"""
        if len(values) < window_size:
            return values
        weights = np.exp(-(np.arange(window_size) - window_size//2)**2/8)
        weights = weights/np.sum(weights)
        smoothed = np.convolve(values, weights, mode='valid')
        # 处理边界情况
        pad_size = (len(values) - len(smoothed))//2
        smoothed = np.pad(smoothed, (pad_size, pad_size), mode='edge')
        return smoothed.tolist()
    
    # 在记录结果后添加平滑处理
    for dataset_name in results:
        results[dataset_name]["pa_smooth"] = smooth_results(results[dataset_name]["pa"])
        results[dataset_name]["ga_smooth"] = smooth_results(results[dataset_name]["ga"])
    
    # 固定输出目录：图和 CSV 都写这里，每次跑完覆盖更新（和窗口实验一样）
    server_save_dir = "/home/zhouzt/experiment/log_parsing/log3p_parsing/top-n experiments"
    os.makedirs(server_save_dir, exist_ok=True)

    # 保存结果时同时保存原始结果和平滑后的结果
    for metric in config.EXPERIMENT["metrics"]:
        df = pd.DataFrame({
            "TOP_N": list(config.EXPERIMENT["top_n_range"])
        })
        
        for dataset in config.EXPERIMENT["datasets"]:
            df[f"{dataset}_original"] = results[dataset][metric]
            df[f"{dataset}_smooth"] = results[dataset][f"{metric}_smooth"]
        
        csv_path = os.path.join(experiment_dirs["results"], f"{metric}_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {metric} results to {csv_path}")
        fixed_csv_path = os.path.join(server_save_dir, f"{metric}_results.csv")
        df.to_csv(fixed_csv_path, index=False)
        logger.info(f"Saved {metric} results to {fixed_csv_path}")
    
    plot_results(results, server_save_dir, use_smoothed=True)
    
    return results

def plot_results(results, save_dir, use_smoothed=True, *, tick_fontsize=20, legend_fontsize=None):
    """绘制实验结果折线图"""
    # 设置中文字体支持（如果需要）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    for metric in config.EXPERIMENT["metrics"]:
        plt.figure(figsize=(12, 8))
        
        for dataset in config.EXPERIMENT["datasets"]:
            values = results[dataset][f"{metric}_smooth"] if use_smoothed else results[dataset][metric]
            plt.plot(
                list(config.EXPERIMENT["top_n_range"]), 
                values,
                label=dataset.upper(),
                marker='o',
                markersize=4,
                linewidth=2
            )
        
        # 仅保留坐标轴刻度（x/y的数值指标），不显示标题和轴名称
        plt.xlabel("")
        plt.ylabel("")
        # 修改坐标轴刻度字号（x/y轴的数字）
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.legend(loc='best', fontsize=(tick_fontsize if legend_fontsize is None else legend_fontsize))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(save_dir, f"{metric}_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {metric} plot to {os.path.join(save_dir, f'{metric}_plot.png')}")

if __name__ == "__main__":
    run_top_n_experiment()