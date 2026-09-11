import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import argparse

def load_templates(file_path):
    """加载模板数据"""
    print(f"加载模板文件: {file_path}")
    df = pd.read_csv(file_path)
    print(f"共加载 {len(df)} 个模板")
    return df

def preprocess_template(template):
    """预处理模板，移除<*>并分割成单词列表"""
    # 将<*>替换为空格，然后分割成单词
    words = re.sub(r'<\*>', ' ', template).split()
    # 移除标点符号
    words = [re.sub(r'[^\w\s]', '', word.lower()) for word in words]
    # 移除空字符串
    words = [word for word in words if word]
    return words

def calculate_word_frequency(templates):
    """计算所有模板中单词的频率"""
    word_counter = Counter()
    for template in templates:
        words = preprocess_template(template)
        word_counter.update(words)
    return word_counter

def calculate_pairwise_overlap(templates):
    """计算模板间的成对重叠度"""
    # 预处理所有模板
    processed_templates = [set(preprocess_template(template)) for template in templates]
    
    results = []
    # 对所有模板对计算Jaccard相似度
    for i, j in combinations(range(len(templates)), 2):
        template_i = processed_templates[i]
        template_j = processed_templates[j]
        
        # 计算Jaccard相似度
        intersection = len(template_i.intersection(template_j))
        union = len(template_i.union(template_j))
        
        if union > 0:
            jaccard = intersection / union
        else:
            jaccard = 0
            
        results.append({
            'Template1': templates[i],
            'Template2': templates[j],
            'Overlap': intersection,
            'Union': union,
            'JaccardSimilarity': jaccard
        })
    
    return pd.DataFrame(results)

def calculate_template_uniqueness(templates):
    """计算每个模板的唯一性"""
    # 使用TF-IDF向量化模板
    vectorizer = TfidfVectorizer(preprocessor=lambda x: ' '.join(preprocess_template(x)))
    tfidf_matrix = vectorizer.fit_transform(templates)
    
    # 计算每个模板的唯一词汇
    unique_words = []
    
    for i, template in enumerate(templates):
        words = preprocess_template(template)
        # 获取该模板的TF-IDF权重
        feature_index = tfidf_matrix[i, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
        
        # 找出高TF-IDF分数的词（唯一性较高的词）
        high_tfidf_words = [(vectorizer.get_feature_names_out()[i], s) for (i, s) in tfidf_scores if s > 0.5]
        
        unique_words.append({
            'Template': template,
            'Words': words,
            'UniqueWords': [w for w, s in high_tfidf_words],
            'UniquenessScore': np.mean([s for w, s in high_tfidf_words]) if high_tfidf_words else 0
        })
    
    return pd.DataFrame(unique_words)

def generate_overlap_heatmap(pairwise_overlap, output_dir):
    """生成重叠度热图"""
    # 构建方阵
    n = int((1 + np.sqrt(1 + 8 * len(pairwise_overlap))) / 2)  # n(n-1)/2 = len
    matrix = np.zeros((n, n))
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            matrix[i, j] = pairwise_overlap.iloc[idx]['JaccardSimilarity']
            matrix[j, i] = matrix[i, j]  # 对称矩阵
            idx += 1
    
    # 绘制热图
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title('Template Pairwise Jaccard Similarity')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'template_overlap_heatmap.png'))
    plt.close()

def process_templates(templates, split_regex=r'\s+'):
    """处理模板，将包含<*>的部分替换为<*>"""
    processed_templates = []
    for template in templates:
        # 分割模板
        template_tokens = re.split(split_regex, template)
        # 处理每个token
        processed_tokens = ['<*>' if '<*>' in token else token for token in template_tokens]
        # 重新组合
        processed_template = ' '.join(processed_tokens)
        processed_templates.append(processed_template)
    return processed_templates

def analyze_templates(file_path, output_dir='./template_analysis'):
    """分析模板并生成报告"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 加载模板
    df = load_templates(file_path)
    original_templates = df['EventTemplate'].tolist()
    
    # 处理模板，将包含<*>的部分替换为<*>
    templates = process_templates(original_templates)
    
    # 打印原始和处理后的模板对比
    print("\n===== 模板处理示例 =====")
    for i, (orig, proc) in enumerate(zip(original_templates, templates)):
        if i < 5:  # 只打印前5个示例
            print(f"原始模板: {orig}")
            print(f"处理后: {proc}")
            print("-" * 50)
    
    # 1. 计算单词频率
    word_freq = calculate_word_frequency(templates)
    word_freq_df = pd.DataFrame({'Word': list(word_freq.keys()), 'Frequency': list(word_freq.values())})
    word_freq_df = word_freq_df.sort_values('Frequency', ascending=False)
    
    # 2. 计算模板间的重叠度
    pairwise_overlap = calculate_pairwise_overlap(templates)
    
    # 3. 计算每个模板的唯一性
    template_uniqueness = calculate_template_uniqueness(templates)
    
    # 4. 生成热图
    generate_overlap_heatmap(pairwise_overlap, output_dir)
    
    # 5. 保存结果
    dataset_name = os.path.basename(file_path).split('.')[0]
    
    # 保存原始与处理后的模板对照表
    template_comparison = pd.DataFrame({
        'OriginalTemplate': original_templates,
        'ProcessedTemplate': templates
    })
    template_comparison.to_csv(os.path.join(output_dir, f'{dataset_name}_processed_templates.csv'), index=False)
    
    word_freq_df.to_csv(os.path.join(output_dir, f'{dataset_name}_word_frequency.csv'), index=False)
    pairwise_overlap.to_csv(os.path.join(output_dir, f'{dataset_name}_pairwise_overlap.csv'), index=False)
    template_uniqueness.to_csv(os.path.join(output_dir, f'{dataset_name}_template_uniqueness.csv'), index=False)
    
    # 6. 计算汇总统计数据
    avg_overlap = pairwise_overlap['JaccardSimilarity'].mean()
    max_overlap = pairwise_overlap['JaccardSimilarity'].max()
    avg_uniqueness = template_uniqueness['UniquenessScore'].mean()
    
    # 7. 打印报告
    print("\n===== 模板分析报告 =====")
    print(f"数据集: {dataset_name}")
    print(f"模板数量: {len(templates)}")
    print(f"总词汇数: {sum(word_freq.values())}")
    print(f"独特词汇数: {len(word_freq)}")
    print(f"平均模板重叠度 (Jaccard): {avg_overlap:.4f}")
    print(f"最大模板重叠度 (Jaccard): {max_overlap:.4f}")
    print(f"平均模板唯一性分数: {avg_uniqueness:.4f}")
    
    print(f"\n最常见的10个词:")
    for word, freq in word_freq.most_common(10):
        print(f"  {word}: {freq}")
    
    print(f"\n重叠度最高的5对模板:")
    top_overlaps = pairwise_overlap.sort_values('JaccardSimilarity', ascending=False).head(5)
    for _, row in top_overlaps.iterrows():
        print(f"  相似度: {row['JaccardSimilarity']:.4f}")
        print(f"    模板1: {row['Template1']}")
        print(f"    模板2: {row['Template2']}")
        print(f"    共同词汇: {row['Overlap']}")
    
    # 保存报告摘要
    with open(os.path.join(output_dir, f'{dataset_name}_report.txt'), 'w') as f:
        f.write("===== 模板分析报告 =====\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"模板数量: {len(templates)}\n")
        f.write(f"总词汇数: {sum(word_freq.values())}\n")
        f.write(f"独特词汇数: {len(word_freq)}\n")
        f.write(f"平均模板重叠度 (Jaccard): {avg_overlap:.4f}\n")
        f.write(f"最大模板重叠度 (Jaccard): {max_overlap:.4f}\n")
        f.write(f"平均模板唯一性分数: {avg_uniqueness:.4f}\n")
    
    print(f"\n分析结果保存在: {output_dir}")
    
    return {
        'word_freq': word_freq_df,
        'pairwise_overlap': pairwise_overlap,
        'template_uniqueness': template_uniqueness,
        'summary': {
            'dataset': dataset_name,
            'template_count': len(templates),
            'total_words': sum(word_freq.values()),
            'unique_words': len(word_freq),
            'avg_overlap': avg_overlap,
            'max_overlap': max_overlap,
            'avg_uniqueness': avg_uniqueness
        }
    }

def compare_datasets(file_paths, output_dir='./template_analysis'):
    """比较多个数据集的模板重叠度"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = []
    for file_path in file_paths:
        result = analyze_templates(file_path, output_dir)
        results.append(result['summary'])
    
    # 创建比较表格
    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(os.path.join(output_dir, 'datasets_comparison.csv'), index=False)
    
    # 绘制比较图表
    plt.figure(figsize=(12, 6))
    
    # 平均重叠度比较
    plt.subplot(1, 2, 1)
    plt.bar(comparison_df['dataset'], comparison_df['avg_overlap'])
    plt.title('Average Template Overlap')
    plt.ylabel('Jaccard Similarity')
    plt.xticks(rotation=45)
    
    # 平均唯一性比较
    plt.subplot(1, 2, 2)
    plt.bar(comparison_df['dataset'], comparison_df['avg_uniqueness'])
    plt.title('Average Template Uniqueness')
    plt.ylabel('Uniqueness Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'datasets_comparison.png'))
    plt.close()
    
    print("\n===== 数据集比较报告 =====")
    print(comparison_df)
    print(f"\n比较结果保存在: {output_dir}")

def extract_diverse_templates(file_path, output_file, target_ratio=0.3, num_templates=20, structured_log_path=None):
    """
    抽取模板，使独特词汇数量约为所有词汇总数（包括重复）的指定比例。
    
    Args:
        file_path: 模板CSV文件路径
        output_file: 输出文件路径
        target_ratio: 目标独特词汇比例 (默认0.3，即30%)
        num_templates: 需要抽取的模板数量 (默认20)
        structured_log_path: 结构化日志文件路径，用于获取日志内容
    
    Returns:
        抽取的模板列表以及相关统计信息
    """
    print(f"从 {file_path} 中抽取 {num_templates} 个模板...")
    print(f"目标独特词汇比例: {target_ratio:.1%}")
    
    # 加载模板
    df = load_templates(file_path)
    all_templates = df['EventTemplate'].tolist()
    
    # 处理模板
    processed_templates = process_templates(all_templates)
    
    # 计算所有模板中的词汇
    all_words_list = []  # 所有词汇（包括重复）
    for template in processed_templates:
        words = preprocess_template(template)
        all_words_list.extend(words)
    
    total_word_count = len(all_words_list)  # 所有词汇数（包括重复）
    target_unique_word_count = int(total_word_count * target_ratio)  # 目标独特词汇数
    
    # 统计原始数据集的独特词汇
    all_unique_words = set(all_words_list)
    unique_word_count = len(all_unique_words)
    
    print(f"所有词汇总数（包括重复）: {total_word_count}")
    print(f"原始数据集独特词汇数: {unique_word_count}")
    print(f"目标独特词汇数（总词汇的{target_ratio:.1%}）: {target_unique_word_count}")
    
    # 为每个模板计算词汇信息
    template_info = []
    for i, template in enumerate(processed_templates):
        words = preprocess_template(template)
        unique_words = set(words)
        template_info.append({
            'index': i,
            'eventid': df.iloc[i].get('EventId', f"event_{i}"),
            'template': all_templates[i],
            'words': words,  # 所有词汇（包括重复）
            'unique_words': unique_words  # 独特词汇
        })
    
    # 尝试多次贪心选择，找到最接近目标独特词汇数的方案
    best_selected = None
    best_diff = float('inf')
    
    for attempt in range(10):
        # 随机排序模板
        import random
        random.seed(attempt)
        candidate_templates = template_info.copy()
        random.shuffle(candidate_templates)
        
        # 贪心选择
        selected = []
        selected_unique_words = set()
        
        while len(selected) < num_templates:
            best_template = None
            best_score = -1
            
            for template in candidate_templates:
                if template in selected:
                    continue
                
                # 计算新增独特词汇
                new_words = template['unique_words'] - selected_unique_words
                new_word_count = len(new_words)
                
                # 如果新增词汇有助于接近目标，选择该模板
                current_count = len(selected_unique_words)
                new_count = current_count + new_word_count
                
                # 计算选择该模板后与目标的接近程度
                current_diff = abs(current_count - target_unique_word_count)
                new_diff = abs(new_count - target_unique_word_count)
                
                # 如果新差异更小，或者当前差异相同但新增词汇更多
                if new_diff < current_diff or (new_diff == current_diff and new_word_count > 0):
                    best_template = template
                    best_score = new_word_count
            
            # 如果找不到更好的模板，随机选择一个未选中的模板
            if best_template is None:
                remaining = [t for t in candidate_templates if t not in selected]
                if remaining:
                    best_template = random.choice(remaining)
                else:
                    break
            
            # 添加选中的模板
            selected.append(best_template)
            selected_unique_words.update(best_template['unique_words'])
            
            # 检查是否已经非常接近目标
            if abs(len(selected_unique_words) - target_unique_word_count) <= 2:
                break
        
        # 如果还没选够指定数量的模板，随机选择剩余模板
        if len(selected) < num_templates:
            remaining = [t for t in candidate_templates if t not in selected]
            random.shuffle(remaining)
            selected.extend(remaining[:num_templates - len(selected)])
        
        # 计算最终选择的独特词汇数
        final_unique_words = set()
        for template in selected[:num_templates]:
            final_unique_words.update(template['unique_words'])
        
        # 计算与目标的差距
        diff = abs(len(final_unique_words) - target_unique_word_count)
        print(f"尝试 {attempt+1}/10: 独特词汇数 = {len(final_unique_words)}, 与目标差距 = {diff}")
        
        # 更新最佳结果
        if diff < best_diff:
            best_diff = diff
            best_selected = selected[:num_templates]
    
    # 使用最佳选择
    selected_templates = best_selected
    
    # 计算最终统计数据
    final_unique_words = set()
    final_all_words = []
    
    for template in selected_templates:
        final_unique_words.update(template['unique_words'])
        final_all_words.extend(template['words'])
    
    # 获取对应的所有日志行
    selected_event_ids = [t['eventid'] for t in selected_templates]
    
    if structured_log_path and os.path.exists(structured_log_path):
        print(f"从 {structured_log_path} 中加载结构化日志...")
        structured_logs = pd.read_csv(structured_log_path)
        
        # 筛选所选模板对应的所有日志行
        selected_logs = structured_logs[structured_logs['EventId'].isin(selected_event_ids)]
        
        # 确保有 LineId 和 Content 列
        if 'LineId' not in selected_logs.columns:
            selected_logs['LineId'] = selected_logs.index + 1
        
        if 'Content' not in selected_logs.columns:
            # 尝试不同可能的列名
            content_column_candidates = ['Content', 'log_content', 'Message', 'Log', 'Content']
            for column in content_column_candidates:
                if column in structured_logs.columns:
                    selected_logs['Content'] = structured_logs[column]
                    print(f"使用列 '{column}' 作为日志内容")
                    break
            else:
                print("警告: 无法找到原始日志内容列，将使用空字符串作为Content")
                selected_logs['Content'] = ""
        
        # 确保只保留必要的列：EventId、EventTemplate、LineId 和 Content
        keep_columns = ['EventId', 'EventTemplate', 'LineId', 'Content']
        keep_columns = [col for col in keep_columns if col in selected_logs.columns]
        
        # 如果缺少 EventTemplate 列，则添加它
        if 'EventTemplate' not in selected_logs.columns:
            # 创建 EventId 到 EventTemplate 的映射
            template_map = {t['eventid']: t['template'] for t in selected_templates}
            selected_logs['EventTemplate'] = selected_logs['EventId'].map(template_map)
        
        # 只保留需要的列
        selected_logs = selected_logs[keep_columns]
        
        # 保存结果
        print(f"找到 {len(selected_logs)} 条对应的日志行")
        
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存到CSV
        selected_logs.to_csv(output_file, index=False)
        print(f"结果已保存到: {output_file}")
    else:
        # 如果没有提供结构化日志路径，或文件不存在，则只保存模板
        print("未提供结构化日志文件或文件不存在，只保存模板信息")
        output_df = pd.DataFrame({
            'EventId': [t['eventid'] for t in selected_templates],
            'EventTemplate': [t['template'] for t in selected_templates]
        })
        
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存到CSV
        output_df.to_csv(output_file, index=False)
        print(f"仅模板结果已保存到: {output_file}")
    
    # 计算最终的词汇比例
    unique_words_ratio = len(final_unique_words) / total_word_count
    
    print(f"\n===== 抽取结果 =====")
    print(f"已选择 {len(selected_templates)} 个模板")
    print(f"所有词汇总数（包括重复）: {total_word_count}")
    print(f"所选模板独特词汇数: {len(final_unique_words)}")
    print(f"独特词汇占总词汇比例: {unique_words_ratio:.2%}")
    
    return {
        'selected_templates': selected_templates,
        'total_words': total_word_count,
        'unique_words': len(final_unique_words),
        'unique_ratio': unique_words_ratio
    }

if __name__ == "__main__":
    # 在这里直接设置文件路径，不需要通过命令行参数提供
    # Zookeeper数据集
    zookeeper_file = "/home/zhouzt/experiment/log_parsing/logs/Mac/Mac_2k.log_templates.csv"
    output_dir = "/home/zhouzt/experiment/log_parsing/logs/Mac/template_analysis"  # 默认输出目录
    
    # 抽取模板并保存到30%文件，并包含原始日志内容
    output_file = "/home/zhouzt/experiment/log_parsing/30%/Mac/selected_templates.csv"
    structured_log_file = "/home/zhouzt/experiment/log_parsing/logs/Mac/Mac_2k.log_structured.csv"
    extract_diverse_templates(zookeeper_file, output_file, target_ratio=0.3, num_templates=150, structured_log_path=structured_log_file)
    

    # 单个数据集分析
    # analyze_templates(zookeeper_file, output_dir)
    
    # 多个数据集比较
    # compare_datasets([hdfs_file, bgl_file, spirit_file], output_dir)
    
    # 如果你仍然想保留命令行参数的功能，可以使用以下代码:
    '''
    parser = argparse.ArgumentParser(description='分析日志模板的单词重叠度')
    parser.add_argument('--files', nargs='+', default=[zookeeper_file], help='模板文件路径列表')
    parser.add_argument('--output', default=output_dir, help='输出目录')
    
    args = parser.parse_args()
    
    if len(args.files) == 1:
        analyze_templates(args.files[0], args.output)
    else:
        compare_datasets(args.files, args.output)
    ''' 