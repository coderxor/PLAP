import re

import pandas as pd

# 读取HDFS_2k.log_structured.csv文件
structured_logs_df = pd.read_csv('/home/zhouzt/experiment/log_parsing/logs/HDFS/HDFS_2k.log_structured.csv')

# 检查数据结构
# print(structured_logs_df.columns)  # 确保包含 LineId, Date, Time, Pid, Level, Component, Content, EventId, EventTemplate

# 1. 按 EventId 进行分组（假设EventId为E1, E2, ...）
grouped = structured_logs_df.groupby('EventId')

# 2. 创建一个列表存储抽样后的日志
Dtrain = []

# 3. 对每个 EventId 分组进行处理，抽样 2 条日志
for event_id, group in grouped:
    n = 1  # 每组抽样的数量

    # 先进行一次抽样
    sampled_logs = group.sample(n=min(n, len(group)), random_state=42, replace=False)  # 防止不够时重复抽样

    # 如果抽样数量小于n，则继续重复抽样直到达到数量n
    while len(sampled_logs) < n:
        # 从该组继续抽样，replace=True表示可以重复抽样
        additional_logs = group.sample(n=n - len(sampled_logs), random_state=42, replace=True)
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
        Dtrain.append((log, template,line_id, event_id))

# 4. 打印 Dtrain 按 EventId 从小到大排序输出
Dtrain = sorted(Dtrain, key=lambda x: int(x[3][1:]))  # 按 EventId 排序 (x[3] 对应 event_id)

# 输出排序后的结果
for log, template, line_id, event_id in Dtrain:
    print(f"EventId:{event_id}, LineId: {line_id}, Original Log: {log}, Template: {template}")

