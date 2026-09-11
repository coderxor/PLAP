import re
import torch
import torch.nn as nn
from tqdm import tqdm
from log_parsing.log3p_parsing.config import DEVICE, get_current_dataset_config

def decode_predictions(predictions, tokenizer):
    """将索引转换回对应的token"""
    tokens = tokenizer.convert_ids_to_tokens(predictions, skip_special_tokens=True)
    return " ".join(tokens).replace(" ##", "")

def remove_wildcards(tokens):
    """移除 <*> 并返回剩余的有效令牌"""
    return [token for token in tokens if token != "<*>"]

def truncate_to_length(tokens, length):
    """将令牌列表截断到指定长度"""
    return tokens[:length]

def eval_model(model, dataloader, tokenizer):
    """
    评估模型性能
    
    Args:
        model: 预训练模型
        dataloader: 测试数据加载器
        tokenizer: 分词器
        
    Returns:
        results: 评估结果列表
        avg_loss: 平均损失
        pa: 解析准确率 (Parsing Accuracy)
        ga: 组准确度 (Group Accuracy)
    """
    model.eval()  # 切换到评估模式
    
    total_loss = 0
    total_correct_logs = 0  # 正确解析的日志消息总数
    total_logs = 0  # 日志消息总数
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    results = []  # 存储测试结果，用于分析
    template_groups = {}  # 存储原始模板的分组
    predicted_groups = {}  # 存储预测模板的分组
    
    def postprocess_prediction(decoded_tokens, log_length):
        """后处理预测结果，将PARAM替换为<*>并处理长度"""
        processed_tokens = []
       
        # 遍历所有的decoded_tokens，将"PARAM"替换为"<*>"
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
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    with torch.no_grad():  # 禁用梯度计算
        for batch in progress_bar:
            logs, input_ids, target_ids, templates, line_ids, event_ids = batch
            
            # 将数据移到GPU
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(DEVICE)
            
            # 模型推理
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 计算Loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item()
            
            # 获取预测结果
            predictions = torch.argmax(logits, dim=-1)
            
            # 逐日志处理
            for original_log, prediction, template, line_id, event_id in zip(logs, predictions, templates, line_ids, event_ids):
                decoded_prediction = decode_predictions(prediction.cpu().numpy(), tokenizer)
                
                # 使用配置文件中的分隔符
                current_config = get_current_dataset_config()
                log_length = len(re.split(current_config["split_regex"], template))  # 获取原始日志的长度
                
                # 替换PARAM并处理长度
                processed_prediction = postprocess_prediction(decoded_prediction.split(), log_length)

                # 处理原始模板
                template_tokens = re.split(current_config["split_regex"], template)
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
                
                # 添加到输出数据
                output_data.append([line_id, original_log, " ".join(processed_prediction)])
    
    # 改进的GA计算
    correct_group_logs = 0
    template_to_pred = {}  # 真实模板到预测模板的映射
    pred_to_template = {}  # 预测模板到真实模板的映射

    # 为每个日志找到对应的真实模板和预测模板
    for original_log, predicted_template, true_template in results:
        if true_template not in template_to_pred:
            template_to_pred[true_template] = set()
        template_to_pred[true_template].add(predicted_template)
        
        if predicted_template not in pred_to_template:
            pred_to_template[predicted_template] = set()
        pred_to_template[predicted_template].add(true_template)

    # 计算一对一映射的日志数量
    for template, logs in template_groups.items():
        # 如果真实模板只映射到一个预测模板，且该预测模板只映射到这个真实模板
        if (template in template_to_pred and 
            len(template_to_pred[template]) == 1 and
            len(pred_to_template[next(iter(template_to_pred[template]))]) == 1):
            correct_group_logs += len(logs)
    
    # 计算最终指标
    ga = correct_group_logs / total_logs if total_logs > 0 else 0  # 组准确度
    avg_loss = total_loss / len(dataloader)  # 平均Loss
    pa = total_correct_logs / total_logs if total_logs > 0 else 0  # 日志级别准确率
    
    return results, avg_loss, pa, ga

def print_evaluation_results(results, avg_loss, pa, ga, num_examples=5):
    """打印评估结果"""
    print("\n===== 评估结果 =====")
    print(f"平均损失 (Loss): {avg_loss:.4f}")
    print(f"解析准确率 (PA): {pa:.2%}")
    print(f"组准确度 (GA): {ga:.2%}")
    
    print(f"\n===== 样例结果 (显示前{num_examples}个) =====")
    for i, (original_log, predicted_template, template) in enumerate(results[:num_examples]):
        print(f"样例 {i + 1}:")
        print(f"原始日志: {original_log}")
        print(f"预测模板: {predicted_template}")
        print(f"原始模板: {template}")
        print("-" * 50)