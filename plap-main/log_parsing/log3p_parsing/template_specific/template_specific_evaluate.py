import re
import torch
import torch.nn as nn
from tqdm import tqdm
from log_parsing.log3p_parsing.template_specific.template_specific_config import DEVICE, get_current_dataset_config

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

def print_template_specific_results(results, avg_loss, template_pa, ga=None, num_examples=5):
    """打印模板特定参数的评估结果"""
    print("\n===== 模板特定参数评估结果 =====")
    print(f"平均损失 (Loss): {avg_loss:.4f}")
    print(f"解析准确率 (PA): {template_pa:.2%}")
    if ga is not None:
        print(f"组准确度 (GA): {ga:.2%}")
    
    print(f"\n===== 样例结果 (显示前{num_examples}个) =====")
    for i, (original_log, predicted_template, template, expected_param, keywords_match, param_match, is_correct) in enumerate(results[:num_examples]):
        print(f"样例 {i + 1}:")
        print(f"原始日志: {original_log}")
        print(f"预测模板: {predicted_template}")
        print(f"原始模板: {template}")
        print(f"期望参数: {expected_param}")
        print(f"关键词匹配: {'✓' if keywords_match else '✗'}")
        print(f"参数匹配: {'✓' if param_match else '✗'}")
        print(f"预测结果: {'✓ 正确' if is_correct else '✗ 错误'}")
        print("-" * 50)

def convert_params_to_wildcards(tokens):
    """将PARAM格式的标记转换为<*>"""
    return ['<*>' if token.startswith('PARAM') else token for token in tokens]

def eval_template_specific_model(model, dataloader, tokenizer, template_to_param_id):
    """评估模板特定参数的模型性能，与evaluate.py保持一致并增加参数特定性判断"""
    model.eval()
    
    total_loss = 0
    total_correct_logs = 0
    total_logs = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    results = []
    template_groups = {}
    predicted_groups = {}
    
    # 存储结果，用于保存到文件
    output_data = []
    
    progress_bar = tqdm(dataloader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in progress_bar:
            logs, input_ids, target_ids, templates, line_ids, event_ids = batch
            
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=-1)
            
            for original_log, prediction, template, line_id, event_id in zip(
                logs, predictions, templates, line_ids, event_ids):
                
                decoded_prediction = decode_predictions(prediction.cpu().numpy(), tokenizer)
                current_config = get_current_dataset_config()
                
                # 处理原始模板，与evaluate.py保持一致
                template_tokens = re.split(current_config["split_regex"], template)
                processed_template = ['<*>' if '<*>' in token else token for token in template_tokens]
                
                # 解析预测结果
                predicted_tokens = decoded_prediction.split()
                
                # 处理预测长度，截断到模板长度
                predicted_tokens = truncate_to_length(predicted_tokens, len(processed_template))
                
                # 第1步：检查关键词是否匹配
                # 将预测结果中的PARAMx转换为<*>用于评估
                predicted_wildcards = convert_params_to_wildcards(predicted_tokens)
                
                # 移除<*>，仅保留关键词进行比较
                filtered_prediction = remove_wildcards(predicted_wildcards)
                filtered_template = remove_wildcards(processed_template)
                
                # 仅关键词的匹配结果
                keywords_match = (filtered_prediction == filtered_template)
                
                # 第2步：检查参数是否与模板ID匹配
                expected_param = f"PARAM{event_id}"  # 期望的参数标记
                param_match = True  # 默认为真，如果发现不匹配则设为假
                
                # 查找模板中的所有参数位置
                param_indices = [i for i, token in enumerate(processed_template) if token == "<*>"]
                
                # 如果模板中没有参数，则参数匹配默认为True
                if len(param_indices) == 0:
                    param_match = True
                # 否则检查所有参数位置
                elif len(predicted_tokens) == len(processed_template):
                    for idx in param_indices:
                        # 检查预测中的参数位置是否为期望的参数
                        if not predicted_tokens[idx].startswith(expected_param):
                            param_match = False
                            break
                else:
                    # 长度不匹配时参数检查无法进行，设为False
                    param_match = False
                
                # 最终结果：关键词匹配 AND 参数匹配
                is_correct = keywords_match and param_match
                
                if is_correct:
                    total_correct_logs += 1
                total_logs += 1
                
                # 更新模板组统计 - 使用处理后的模板
                template_key = " ".join(processed_template)
                if template_key not in template_groups:
                    template_groups[template_key] = []
                template_groups[template_key].append(original_log)
                
                # 更新预测模板组
                prediction_key = " ".join(predicted_wildcards)
                if prediction_key not in predicted_groups:
                    predicted_groups[prediction_key] = []
                predicted_groups[prediction_key].append(original_log)
                
                # 保存结果 - 使用处理后的模板，并包含参数匹配信息
                results.append((
                    original_log, 
                    " ".join(predicted_tokens),        # 原始预测（带PARAMx）
                    " ".join(processed_template),      # 处理后的模板
                    expected_param,                    # 期望的参数
                    keywords_match,                    # 关键词是否匹配
                    param_match,                       # 参数是否匹配
                    is_correct                         # 整体是否正确
                ))
                
                # 添加到输出数据
                output_data.append([line_id, original_log, " ".join(predicted_wildcards)])
    
    # 计算GA指标 - 使用evaluate.py中相同的方法
    correct_group_logs = 0
    template_to_pred = {}  # 真实模板到预测模板的映射
    pred_to_template = {}  # 预测模板到真实模板的映射

    # 为每个日志找到对应的真实模板和预测模板
    for original_log, predicted_template, template, expected_param, keywords_match, param_match, is_correct in results:
        if template not in template_to_pred:
            template_to_pred[template] = set()
        template_to_pred[template].add(predicted_template)
        
        if predicted_template not in pred_to_template:
            pred_to_template[predicted_template] = set()
        pred_to_template[predicted_template].add(template)

    # 计算一对一映射的日志数量
    for template, logs in template_groups.items():
        # 如果真实模板只映射到一个预测模板，且该预测模板只映射到这个真实模板
        if (template in template_to_pred and 
            len(template_to_pred[template]) == 1 and
            len(pred_to_template[next(iter(template_to_pred[template]))]) == 1):
            correct_group_logs += len(logs)
    
    # 计算最终指标
    ga = correct_group_logs / total_logs if total_logs > 0 else 0  # 组准确度
    template_pa = total_correct_logs / total_logs if total_logs > 0 else 0  # 整体准确率（关键词+参数）
    avg_loss = total_loss / len(dataloader)
    
    return results, avg_loss, template_pa, ga, output_data 

