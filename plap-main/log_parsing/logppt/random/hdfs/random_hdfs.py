import random
import textdistance
import re
import pandas as pd

# 停用词列表，可以根据需求进行扩展
stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'of', 'for', 'to',
              'with', 'without', 'as'}


# 将驼峰命名法转换为用空格分隔的单词
def split_camel_case(log):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', log)


# 日志预处理函数，处理驼峰命名法，去掉停用词和非字符符号，但保留原始日志结构
def preprocess_log(log):
    # 1. 处理驼峰命名法
    log = split_camel_case(log)

    # 2. 移除非字符符号，仅保留字母、数字和空格
    log = re.sub(r'[^a-zA-Z0-9\s]', '', log)

    # 3. 保留原始格式，处理停用词
    log_words = log.split()
    cleaned_log = ' '.join([word for word in log_words if word.lower() not in stop_words])
    cleaned_log = re.sub(r'\s+', ' ', cleaned_log).strip()
    return cleaned_log


# 定义各种距离函数
def lcs_distance(x, y):
    seq1 = x.split()
    seq2 = y.split()
    lengths = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if seq1[i] == seq2[j]:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    return 1 - 2 * lengths[-1][-1] / (len(seq1) + len(seq2))


def lev_distance(x, y):
    return textdistance.levenshtein.normalized_distance(x, y)


def euc_distance(x, y):
    return textdistance.cosine.normalized_distance(x, y)


def jaccard_distance(x, y):
    return textdistance.jaccard.normalized_distance(x.split(), y.split())


def ratcliff_distance(x, y):
    return textdistance.ratcliff_obershelp.normalized_distance(x, y)


# 最小距离函数，接收距离函数作为参数
def min_distance(c_set, t_set, distance_func):
    D = []
    for c_inst in c_set:
        min_candidate_distance = float('inf')
        for t_inst in t_set:
            min_candidate_distance = min(min_candidate_distance, distance_func(c_inst, t_inst))
        D.append(min_candidate_distance)
    return D


# 自适应随机采样算法
def adaptive_random_sampling(logs, K, eta, preprocessed_logs_df):
    # 第1步：对所有日志进行预处理
    L = [(log, idx) for idx, log in enumerate(logs)]

    # 第2步：初始化Dtrain和S集合，S初始包含最短的日志
    Dtrain = []
    S = [min(L, key=lambda x: len(preprocessed_logs_df.loc[x[1], 'Content']))]  # 按tokens长度选择最短日志

    # 第3步：开始迭代，直到选择K个样本
    while len(S) < K:
        C = []
        while len(C) < eta:
            candidate = random.choice(L)
            if candidate[0] not in [c[0] for c in C] and candidate[0] not in [s[0] for s in S]:
                C.append(candidate)

        # 从候选日志中提取tokens
        candidate_texts = [preprocessed_logs_df.loc[c[1], 'Content'] for c in C]
        candidate_distances = min_distance(candidate_texts, [preprocessed_logs_df.loc[s[1], 'Content'] for s in S],
                                           jaccard_distance)
        best_candidate_index = max(range(len(candidate_distances)), key=candidate_distances.__getitem__)
        selected_candidate = C[best_candidate_index]

        S.append(selected_candidate)
        L.remove(selected_candidate)

    # 将选中的日志、模板和LineId添加到Dtrain
    for original_log, idx in S:
        template = get_template_from_user(idx, preprocessed_logs_df, structured_logs_df)
        line_id = preprocessed_logs_df.loc[idx, 'LineId']  # 获取LineId
        Dtrain.append((original_log, template, line_id))  # 将LineId加入到Dtrain中

    return Dtrain


def get_template_from_user(log_idx, preprocessed_logs_df, structured_logs_df):
    # 根据 LineId 关联模板
    line_id = preprocessed_logs_df.loc[log_idx, 'LineId']
    template = structured_logs_df.loc[structured_logs_df['LineId'] == line_id, 'EventTemplate'].values[0]

    # 将模板按空格和分隔符分开，并替换带有 <*> 的部分
    # token_pattern = r'[^\s,!?=]+'
    template_tokens = re.split(r'[ ,!?=]+', template)
    processed_template = ['<*>' if '<*>' in token else token for token in template_tokens]

    return ' '.join(processed_template)


# 读取数据
preprocessed_logs_df = pd.read_csv('/home/zhouzt/experiment/log_parising/logs/HDFS/HDFS_2k_preprocessed.csv')
structured_logs_df = pd.read_csv('/home/zhouzt/experiment/log_parising/logs/HDFS/HDFS_2k.log_structured.csv')

logs = preprocessed_logs_df['Content'].tolist()
K = 512  # 选择样本数
eta = 32

Dtrain = adaptive_random_sampling(logs, K, eta, preprocessed_logs_df)

# 输出最终的Dtrain
for log, template, line_id in Dtrain:
    print(f" LineId: {line_id}, Original Log: {log}, Template: {template}")
