import re
import pandas as pd


def sample_logs_by_event(file_path, n=1, random_state=42):
    # 读取 CSV 文件
    structured_logs_df = pd.read_csv(file_path)

    # 按 EventId 进行分组
    grouped = structured_logs_df.groupby('EventId')

    # 创建一个列表存储抽样后的日志
    Dtrain = []

    # 对每个 EventId 分组进行处理
    for event_id, group in grouped:
        # 进行一次抽样
        sampled_logs = group.sample(n=min(n, len(group)), random_state=random_state, replace=False)

        # 如果抽样数量小于 n，则继续重复抽样直到达到数量n
        while len(sampled_logs) < n:
            additional_logs = group.sample(n=n - len(sampled_logs), random_state=random_state, replace=True)
            sampled_logs = pd.concat([sampled_logs, additional_logs], ignore_index=True)

        # 将抽样的日志添加到 Dtrain 列表中
        for idx, row in sampled_logs.iterrows():
            line_id = row['LineId']
            log = row['Content']
            event_template = row['EventTemplate']
            template_tokens = re.split(r'[ ,!?=]+', event_template)
            processed_template = ['<*>' if '<*>' in token else token for token in template_tokens]
            template = ' '.join(processed_template)
            event_id = row['EventId']
            Dtrain.append((log, template, line_id, event_id))

    # 按 EventId 从小到大排序
    Dtrain = sorted(Dtrain, key=lambda x: int(x[3][1:]))  # x[3] 对应 event_id

    return Dtrain


def print_sampled_logs(Dtrain):
    for log, template, line_id, event_id in Dtrain:
        print(f"EventId:{event_id}, LineId: {line_id}, Original Log: {log}, Template: {template}")


# 使用示例
file_path = '/home/zhouzt/experiment/log_parising/logs/BGL/BGL_2k.log_structured.csv'
n = 3  # 每组抽样数量
Dtrain = sample_logs_by_event(file_path, n=n)  # 获取抽样后的日志数据
print_sampled_logs(Dtrain)  # 打印处理后的日志
