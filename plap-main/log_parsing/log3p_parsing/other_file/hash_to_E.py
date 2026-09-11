import pandas as pd
from tqdm import tqdm

# 读取两个CSV文件
print("正在读取CSV文件...")
templates_df = pd.read_csv('/home/zhouzt/experiment/log_anomaly_prediction/logparser/Drain/demo_result/BGL.log_templates.csv')
structured_df = pd.read_csv('/home/zhouzt/experiment/log_anomaly_prediction/logprompt/bgl_id_template_embeddings.csv')

# 创建哈希值到行号的映射
print("正在创建哈希值到行号的映射...")
hash_to_row = {}
for index, row in tqdm(templates_df.iterrows(), total=len(templates_df), desc="处理模板文件"):
    # 假设EventId列存储哈希值
    hash_value = row['EventId']
    # 行号从1开始，而不是从0开始
    row_number = index + 1
    hash_to_row[hash_value] = row_number

# 创建一个新列来存储转换后的EventId
print("正在转换EventId...")
tqdm.pandas(desc="处理结构化日志")
structured_df['NewEventId'] = structured_df['EventId'].progress_apply(
    lambda x: f'E{hash_to_row[x]}' if x in hash_to_row else x
)

# 可选：替换原有的EventId列
print("正在更新数据集...")
structured_df['EventId'] = structured_df['NewEventId']
structured_df.drop(columns=['NewEventId'], inplace=True)

# 保存修改后的数据
print("正在保存转换后的数据...")
# 指定一个绝对路径保存结果
output_path = '/home/zhouzt/experiment/log_anomaly_prediction/logprompt/bgl_id_template_embeddings_new.csv'
structured_df.to_csv(output_path, index=False)

print(f"转换完成！结果已保存到: {output_path}")
