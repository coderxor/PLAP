import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from log_anomaly_prediction.logprompt.log3p_prediction.config import *


def load_and_process_bgl_data(log_file_path, window_duration, slide_interval,
                              next_window_duration, interval, n_rows):
    """
    加载和预处理BGL日志数据，生成时间窗口和标签

    Args:
        log_file_path: 日志文件路径
        window_duration: 窗口持续时间
        slide_interval: 滑动间隔
        next_window_duration: 预测窗口持续时间
        interval: 当前窗口与预测窗口之间的间隔
        n_rows: 读取的行数

    Returns:
        texts: 处理后的文本列表
        labels: 对应的标签列表
    """
    # 读取指定数量的日志
    df_log = pd.read_csv(log_file_path, usecols=['Label', 'Timestamp', 'EventId', 'EventTemplate', 'ParameterList'], nrows=n_rows)

    # 将 Unix 时间戳保留为整数值，便于后续计算
    df_log['Timestamp'] = df_log['Timestamp'].astype(int)

    # 处理 Label 列
    def determine_label(label):
        return 0 if label == '-' else 1

    df_log['Label'] = df_log['Label'].apply(determine_label)

    # 排序数据
    df_log = df_log.sort_values(by='Timestamp').reset_index(drop=True)

    texts = []
    labels = []

    start_time_seconds = df_log.iloc[0]['Timestamp']
    while start_time_seconds < df_log['Timestamp'].max():
        # 当前窗口的结束时间
        end_time_seconds = start_time_seconds + window_duration

        # 被预测窗口的开始时间和结束时间
        next_window_start_time_seconds = end_time_seconds + interval
        next_window_end_time_seconds = next_window_start_time_seconds + next_window_duration

        # 检查被预测窗口是否超出了日志的时间范围，如果超出则停止处理
        if next_window_start_time_seconds >= df_log['Timestamp'].max():
            print("没有足够的日志数据生成下一个被预测窗口，停止处理。")
            break

        # 选择窗口内的日志条目
        window_df = df_log[(df_log['Timestamp'] >= start_time_seconds) & (df_log['Timestamp'] < end_time_seconds)]

        # 选择被预测窗口内的日志条目
        next_window_df = df_log[(df_log['Timestamp'] >= next_window_start_time_seconds) & (
                df_log['Timestamp'] < next_window_end_time_seconds)]

        # 如果窗口内的日志条目为0，则跳过
        if len(window_df) == 0:
            start_time_seconds = end_time_seconds
            continue

        # 如果被预测窗口内的日志条目为0，则跳过
        if len(next_window_df) == 0:
            start_time_seconds = next_window_start_time_seconds
            continue

        combined_event_id = ""
        combined_content = ""
        combined_label = 0
        combined_times = []
        combined_params = []

        for _, row in window_df.iterrows():
            event_id = row['EventId'].split()[0]
            content = row['EventTemplate']
            
            param_list = eval(row['ParameterList'])
            param_str = ' '.join(param_list)
            
            combined_event_id += event_id + " "
            combined_content += " ".join(content.split()) + " "
            combined_label = max(combined_label, row['Label'])
            combined_times.append(row['Timestamp'] - start_time_seconds)
            combined_params.append(param_str)

        # 清理文本
        combined_event_id = combined_event_id.strip()
        combined_content = combined_content.strip()
        combined_content = re.sub(r'[^a-zA-Z0-9]', ' ', combined_content)

        # 生成文本格式
        combined_time = " ".join(map(str, combined_times))

        # 获取事件ID (假设已有变量event_id)
        template_id = event_id[1:] if isinstance(event_id, str) and event_id.startswith('E') else event_id

        # 简单地使用模板ID作为参数标识
        template_specific_param = f"template_{template_id}"

        # 根据是否使用提示词决定文本格式
        if USE_PROMPT:
            combined_text = f"semantic {combined_content} sequential {combined_event_id} time {combined_time} param {template_specific_param}"
        else:
            # 无提示词版本：直接拼接内容，不使用关键字
            combined_text = f"{combined_content} {combined_event_id} {combined_time} {template_specific_param}"

        combined_text = ' '.join(combined_text.split())

        texts.append(combined_text)

        # 为预测窗口分配被预测窗口的标签
        combined_label = max(next_window_df['Label'])
        labels.append(combined_label)

        # 更新窗口的开始时间为当前窗口的结束时间
        start_time_seconds = start_time_seconds + slide_interval

    return texts, labels


def load_and_process_hdfs_data(log_file_path, label_file_path, window_duration, slide_interval,
                              next_window_duration, interval, n_rows):
    """
    加载和预处理HDFS日志数据，生成时间窗口和标签

    Args:
        log_file_path: 日志文件路径
        label_file_path: 标签文件路径
        window_duration: 窗口持续时间
        slide_interval: 滑动间隔
        next_window_duration: 预测窗口持续时间
        interval: 当前窗口与预测窗口之间的间隔
        n_rows: 读取的行数

    Returns:
        texts: 处理后的文本列表
        labels: 对应的标签列表
    """
    # 读取日志数据
    df_log = pd.read_csv(log_file_path,
                         usecols=['Date', 'Time', 'EventId', 'Content', 'EventTemplate', 'ParameterList'], nrows=n_rows)
    
    # 读取标签数据并映射
    df_label = pd.read_csv(label_file_path)
    label_mapping = {'Normal': 0, 'Anomaly': 1}
    df_label['Label'] = df_label['Label'].map(label_mapping)

    # 创建从块ID到标签的映射字典
    label_dict = df_label.set_index('BlockId')['Label'].to_dict()

    # 使用内容字段找到对应的标签
    def find_label(content):
        for block_id, label in label_dict.items():
            if block_id in content:
                return label
        return 0

    df_log['Label'] = df_log['Content'].apply(find_label)

    # 将 'Date' 列格式化为字符串，确保前导零保留
    df_log['Date'] = df_log['Date'].apply(lambda x: f"{x:06d}")

    # 合并 'Date' 和 'Time' 列
    df_log['DateTime_str'] = df_log['Date'].astype(str) + ' ' + df_log['Time'].astype(str)

    # 将合并后的字符串转换为 datetime 对象
    df_log['DateTime'] = pd.to_datetime(df_log['DateTime_str'], format='%y%m%d %H%M%S', errors='coerce')

    # 将 'DateTime' 转换为 Unix 时间戳（以秒为单位）
    df_log['Timestamp'] = (df_log['DateTime'] - pd.Timestamp("1970-01-01")).dt.total_seconds()

    # 删除含有 NaT 的行
    df_log = df_log.dropna(subset=['Timestamp'])

    # 排序数据按 'Timestamp' 以便于后续操作
    df_log = df_log.sort_values(by='Timestamp').reset_index(drop=True)

    texts = []
    labels = []

    start_time_seconds = df_log.iloc[0]['Timestamp']
    while start_time_seconds < df_log['Timestamp'].max():
        # 当前窗口的结束时间
        end_time_seconds = start_time_seconds + window_duration

        # 被预测窗口的开始时间和结束时间
        next_window_start_time_seconds = end_time_seconds + interval
        next_window_end_time_seconds = next_window_start_time_seconds + next_window_duration

        # 检查被预测窗口是否超出了日志的时间范围，如果超出则停止处理
        if next_window_start_time_seconds >= df_log['Timestamp'].max():
            print("没有足够的日志数据生成下一个被预测窗口，停止处理。")
            break

        # 选择窗口内的日志条目
        window_df = df_log[(df_log['Timestamp'] >= start_time_seconds) & (df_log['Timestamp'] < end_time_seconds)]

        # 选择被预测窗口内的日志条目
        next_window_df = df_log[(df_log['Timestamp'] >= next_window_start_time_seconds) & (
                df_log['Timestamp'] < next_window_end_time_seconds)]

        # 如果窗口内的日志条目为0，则跳过
        if len(window_df) == 0:
            start_time_seconds = end_time_seconds
            continue

        # 如果被预测窗口内的日志条目为0，则跳过
        if len(next_window_df) == 0:
            start_time_seconds = next_window_start_time_seconds
            continue

        combined_event_id = ""
        combined_content = ""
        combined_label = 0
        combined_times = []
        combined_params = []

        for _, row in window_df.iterrows():
            event_id = row['EventId'].split()[0]
            content = row['EventTemplate']
            
            param_list = eval(row['ParameterList'])
            param_str = ' '.join(param_list)
            
            combined_event_id += event_id + " "
            combined_content += " ".join(content.split()) + " "
            combined_label = max(combined_label, row['Label'])
            combined_times.append(row['Timestamp'] - start_time_seconds)
            combined_params.append(param_str)

        # 清理文本
        combined_event_id = combined_event_id.strip()
        combined_content = combined_content.strip()
        combined_content = re.sub(r'[^a-zA-Z0-9]', ' ', combined_content)
        combined_param = ' '.join(combined_params).strip()

        # 生成文本格式
        combined_time = " ".join(map(str, combined_times))
        
        # 获取事件ID (假设已有变量event_id)
        template_id = event_id[1:] if isinstance(event_id, str) and event_id.startswith('E') else event_id

        # 简单地使用模板ID作为参数标识
        template_specific_param = f"template_{template_id}"

        # 根据是否使用提示词决定文本格式
        if USE_PROMPT:
            combined_text = f"semantic {combined_content} sequential {combined_event_id} time {combined_time} param {template_specific_param}"
        else:
            # 无提示词版本：直接拼接内容，不使用关键字
            combined_text = f"{combined_content} {combined_event_id} {combined_time} {template_specific_param}"

        combined_text = ' '.join(combined_text.split())

        texts.append(combined_text)

        # 为预测窗口分配被预测窗口的标签
        combined_label = max(next_window_df['Label'])
        labels.append(combined_label)

        # 更新窗口的开始时间为当前窗口的结束时间
        start_time_seconds = start_time_seconds + slide_interval

    return texts, labels


def load_and_process_data():
    """
    根据配置加载和处理相应的数据集
    
    Returns:
        texts: 处理后的文本列表
        labels: 对应的标签列表
    """
    if DATASET == 'BGL':
        return load_and_process_bgl_data(
            LOG_FILE_PATH,
            window_duration=WINDOW_DURATION,
            slide_interval=SLIDE_INTERVAL,
            next_window_duration=NEXT_WINDOW_DURATION,
            interval=INTERVAL,
            n_rows=N_ROWS
        )
    elif DATASET == 'HDFS':
        return load_and_process_hdfs_data(
            LOG_FILE_PATH,
            label_file_path=LABEL_FILE_PATH,
            window_duration=WINDOW_DURATION,
            slide_interval=SLIDE_INTERVAL,
            next_window_duration=NEXT_WINDOW_DURATION,
            interval=INTERVAL,
            n_rows=N_ROWS
        )
    else:
        raise ValueError(f"不支持的数据集: {DATASET}")


def split_dataset(texts, labels):
    """
    分割数据集为训练集和测试集

    Args:
        texts: 文本列表
        labels: 标签列表

    Returns:
        训练集和测试集的文本和标签
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    # 打印数据集统计信息
    print_dataset_stats(labels, y_train, y_test)

    return X_train, X_test, y_train, y_test


def print_dataset_stats(all_labels, train_labels, test_labels):
    """打印数据集标签分布统计信息"""
    # 计算每个标签的数量
    all_label_counts = pd.Series(all_labels).value_counts()
    train_label_counts = pd.Series(train_labels).value_counts()
    test_label_counts = pd.Series(test_labels).value_counts()

    print("原始数据集标签分布：")
    print(all_label_counts.to_dict())

    print("训练集标签分布:")
    print(train_label_counts.to_dict())

    print("测试集标签分布:")
    print(test_label_counts.to_dict())