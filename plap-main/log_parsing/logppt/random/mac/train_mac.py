import os
import re

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from logppt.random.mac.random_mac import Dtrain
from virtual_mac import updated_model, updated_tokenizer, add_new_tokens
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LogDataset(Dataset):
    def __init__(self, dtrain, tokenizer, topk=5):
        self.dtrain = dtrain
        self.tokenizer = tokenizer
        self.vocab_set = set(tokenizer.get_vocab().keys())  # 缓存词汇表，提升查询效率
        self.topk = topk

    def __len__(self):
        return len(self.dtrain)

    def __getitem__(self, idx):
        log, template, line_ids = self.dtrain[idx]

        input_tokens = re.split(r'[ ,\[\]]+', log)
        # 动态扩充词汇表
        new_tokens = [token for token in input_tokens if token not in tokenizer.get_vocab()]
        if new_tokens:
            add_new_tokens(input_tokens, self.tokenizer, model)

        template_tokens = re.split(r'[ ,\[\]]+', template)
        param_indices = [i for i, token in enumerate(template_tokens) if token == "<*>"]

        # 创建目标序列：非参数位置与输入相同，参数位置替换为 "PARAM"
        target_tokens = input_tokens.copy()
        for idx in param_indices:
            target_tokens[idx] = "PARAM"

        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

        return log, input_ids, target_ids, template, line_ids


# 初始化分词器和模型
tokenizer = updated_tokenizer
model = updated_model
model = model.to(device)


def collate_fn(batch, device):
    logs, input_ids, target_ids, templates, line_ids = zip(*batch)

    # 将数据移到 GPU
    input_ids = pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True,
                             padding_value=tokenizer.pad_token_id).to(device)
    target_ids = pad_sequence([torch.tensor(ids) for ids in target_ids], batch_first=True,
                              padding_value=tokenizer.pad_token_id).to(device)
    return logs, input_ids, target_ids, templates, line_ids


# 读取文件到 DataFrame
train_file_path = '/logs/Mac/Mac_2k.log_structured.csv'
train_df = pd.read_csv(train_file_path)

# 假设文件有 'Original Log' 和 'Template' 两列
sampled_df = train_df[['Content', 'EventTemplate', 'LineId']].sample(n=2000, random_state=42)
logs = list(zip(sampled_df['Content'], sampled_df['EventTemplate'], sampled_df['LineId']))

# 准备训练数据集和数据加载器
train_dataset = LogDataset(Dtrain, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: collate_fn(x, device))
print(len(train_dataset))  # 检查数据集的大小

# 准备测试数据集和数据加载器
test_dataset = LogDataset(logs, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: collate_fn(x, device))


def train_model(model, dataloader, optimizer, tokenizer):
    model.train()  # 切换到训练模式

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # 忽略填充部分

    total_loss = 0
    total_correct_logs = 0  # 正确解析的日志总数
    total_logs = 0  # 总日志数

    for epoch in range(epochs):  # 训练时包含 epochs 循环
        for batch_idx, batch in enumerate(dataloader):
            logs, input_ids, target_ids, templates, line_ids = batch

            # 将数据移到 GPU 上
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            # 将输入传递给模型，得到 logits
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            logits = outputs.logits  # 获取输出的 logits

            # 计算损失
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            # 累计损失
            total_loss += loss.item()

            # 更新梯度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 批量计算日志准确率
            predicted_ids = logits.argmax(dim=-1)
            valid_predictions = [
                [t for t, m in zip(pred, mask) if m != 0]
                for pred, mask in zip(predicted_ids.tolist(), attention_mask.tolist())
            ]
            valid_targets = [
                [t for t, m in zip(target, mask) if m != 0]
                for target, mask in zip(target_ids.tolist(), attention_mask.tolist())
            ]
            batch_correct_logs = sum(1 for pred, target in zip(valid_predictions, valid_targets) if pred == target)
            total_correct_logs += batch_correct_logs
            total_logs += len(valid_targets)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)  # 平均损失
    accuracy = total_correct_logs / total_logs if total_logs > 0 else 0  # 日志级别的准确率

    return avg_loss, accuracy


# 假设 dataloader 和 optimizer 已经定义，tokenizer 也是可用的
# 训练模型
epochs = 50
best_accuracy = 0.0
best_model_path = None

# 设置优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    avg_loss, accuracy = train_model(model, train_dataloader, optimizer, tokenizer)
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # 更新最佳模型
    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        if best_model_path:  # 如果已有最佳模型，删除旧模型
            os.remove(best_model_path)
        best_model_path = f"best_model_acc_{accuracy:.4f}.pt"
        torch.save(model.state_dict(), best_model_path)  # 保存最优模型

print(f"Best model saved at: {best_model_path} with accuracy: {best_accuracy:.4f}")

# 加载最佳模型权重
if best_model_path:
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path}")


# 将索引转换回对应的 token
def decode_predictions(predictions, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(predictions, skip_special_tokens=True)
    return " ".join(tokens).replace(" ##", "")


def remove_wildcards(tokens):
    # 移除 <*> 并返回剩余的有效令牌
    return [token for token in tokens if token != "<*>"]


def truncate_to_length(tokens, length):
    # 将令牌列表截断到指定长度
    return tokens[:length]


# 测试函数
def eval_model(model, dataloader, tokenizer):
    total_loss = 0
    total_correct_logs = 0  # 正确解析的日志消息总数
    total_logs = 0  # 日志消息总数
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    results = []  # 存储测试结果，用于分析
    template_groups = {}  # 存储原始模板的分组
    predicted_groups = {}  # 存储预测模板的分组

    def postprocess_prediction(decoded_tokens, log_length):
        processed_tokens = []

        # 遍历所有的 decoded_tokens，将 "PARAM" 替换为 "<*>"
        for token in decoded_tokens:
            if token == "PARAM":
                processed_tokens.append("<*>")
            else:
                processed_tokens.append(token)

        # 根据日志长度进行截断
        processed_tokens = truncate_to_length(processed_tokens, log_length)

        return processed_tokens

    # 存储结果，用于保存到文件
    output_data = []

    with torch.no_grad():  # 禁用梯度计算
        for batch in dataloader:
            logs, input_ids, target_ids, templates, line_ids = batch  # 假设 `line_ids` 存储的是日志的 LineId

            # 将数据移到 GPU
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            # 模型推理
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # 计算 Loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item()

            # 获取预测结果
            predictions = torch.argmax(logits, dim=-1)

            # 逐日志处理
            for original_log, prediction, template, line_id in zip(logs, predictions, templates, line_ids):
                decoded_prediction = decode_predictions(prediction.cpu().numpy(), tokenizer)
                log_length = len(re.split(r'[ ,\[\]]+', template))  # 获取原始日志的长度
                # 替换 PARAM 并合并连续的 <*>
                processed_prediction = postprocess_prediction(decoded_prediction.split(), log_length)
                template_tokens = re.split(r'[ ,\[\]]+', template)
                processed_template = ['<*>' if '<*>' in token else token for token in template_tokens]

                # 更新原始模板组
                template_key = " ".join(processed_template)
                if template_key not in template_groups:
                    template_groups[template_key] = []
                template_groups[template_key].append(original_log)

                # 更新预测模板组
                prediction_key = " ".join(processed_prediction)
                if prediction_key not in predicted_groups:
                    predicted_groups[prediction_key] = []
                predicted_groups[prediction_key].append(original_log)

                # 计算日志级别的正确性
                if remove_wildcards(processed_prediction) == remove_wildcards(processed_template):
                    total_correct_logs += 1

                total_logs += 1

                # 记录结果
                results.append((original_log, " ".join(processed_prediction), " ".join(processed_template)))

                # 添加到输出数据，按顺序保存 LineId、原始日志文本、预测模板向量
                output_data.append([line_id, original_log, " ".join(processed_prediction)])

        # 计算 GA（组准确度）
        correct_group_logs = 0
        for template_key, log_group in template_groups.items():
            # 获取该组的预测模板
            predicted_group_keys = {predicted_group for predicted_group, logs in predicted_groups.items() if
                                    any(log in log_group for log in logs)}

            # 如果一个模板组对应唯一的预测组，则认为该组是正确的
            if len(predicted_group_keys) == 1:
                correct_group_logs += len(log_group)

        ga = correct_group_logs / total_logs if total_logs > 0 else 0  # 组准确度
        avg_loss = total_loss / len(dataloader)  # 平均 Loss
        pa = total_correct_logs / total_logs if total_logs > 0 else 0  # 日志级别准确率

        return results, avg_loss, pa, ga


# 执行测试
results, avg_loss, pa, ga = eval_model(model, test_dataloader, tokenizer)

# 打印测试结果
print("测试结果：")
for i, (original_log, predicted_template, template) in enumerate(results):
    print(f"日志 {i + 1}:")
    print(f"原始日志: {original_log}")
    print(f"预测模板: {predicted_template}")
    print(f"原始模板: {template}")
    print("-" * 50)

# 打印总体的 Loss、PA 和 GA
print(f"平均 Loss: {avg_loss:.4f}")
print(f"解析准确率（PA）: {pa:.2%}")
print(f"组准确度（GA）: {ga:.2%}")
