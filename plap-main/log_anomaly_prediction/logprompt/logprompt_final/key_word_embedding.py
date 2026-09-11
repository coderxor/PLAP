from gensim.models import KeyedVectors
import csv

# 加载预训练的Word2Vec模型，例如Google News
# 注意：这个模型需要从网上下载，大小约为1.5GB
# model = KeyedVectors.load_word2vec_format('/path/to/GoogleNews-vectors-negative300.bin', binary=True)

# 示例：加载一个预训练的Word2Vec模型（这里需要提供实际的模型路径）
model = KeyedVectors.load_word2vec_format('/home/zhouzt/code/GoogleNews-vectors-negative300.bin/GoogleNews-vectors'
                                          '-negative300.bin', binary=True)

# 定义你感兴趣的词汇
keywords = ['semantic', 'sequential', 'time']

# 获取词汇表和对应的 embedding 向量
vectors = [model[word] for word in keywords if word in model]

# 保存到 CSV 文件
with open('/home/zhouzt/code/word_embeddings.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头
    writer.writerow(['keyword', 'embedding'])
    # 写入每个单词及其对应的 embedding 向量
    for word, vector in zip(keywords, vectors):
        # 将嵌入向量转换为字符串，使用空格分隔
        vector_str = ' '.join(map(str, vector))
        writer.writerow([word, vector_str])


