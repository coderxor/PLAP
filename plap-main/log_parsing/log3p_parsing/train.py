import os
import torch
import torch.nn as nn
from tqdm import tqdm
from log_parsing.log3p_parsing.config import DEVICE

def train_model(model, dataloader, epochs, optimizer, tokenizer, save_dir='./models', dataset_name='default'):
    """
    训练模型并保存最佳模型和分词器
    
    Args:
        model: 预训练模型
        dataloader: 训练数据加载器
        epochs: 训练轮数
        optimizer: 优化器
        tokenizer: 分词器
        save_dir: 模型保存目录
        dataset_name: 数据集名称，用于命名保存的模型
        
    Returns:
        best_model_path: 最佳模型的保存路径
    """
    model.train()  # 切换到训练模式
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # 忽略填充部分
    
    best_accuracy = 0.0
    best_model_path = None
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
        
        total_loss = 0
        total_correct_logs = 0  # 正确解析的日志总数
        total_logs = 0  # 总日志数
        
        for batch_idx, batch in enumerate(progress_bar):
            logs, input_ids, target_ids, templates, line_ids, event_ids = batch
            
            # 确保数据在正确的设备上
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(DEVICE)
            
            # 将输入传递给模型，得到 logits
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            logits = outputs.logits  # 获取输出的 logits
            
            # 计算损失
            loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
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
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                'accuracy': f"{batch_correct_logs / len(valid_targets):.4f}" if len(valid_targets) > 0 else "N/A"
            })
        
        # 计算本轮的平均损失和准确率
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct_logs / total_logs if total_logs > 0 else 0
        
        print(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # 更新最佳模型
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            
            # 删除旧的最佳模型
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                
                # 也删除旧的tokenizer
                tokenizer_path = os.path.join(save_dir, f"tokenizer_{dataset_name}")
                if os.path.exists(tokenizer_path):
                    import shutil
                    shutil.rmtree(tokenizer_path)
            
            # 保存新的最佳模型
            save_dir = '/home/zhouzt/experiment/log_parsing/log3p_parsing/models'
            # 确保保存目录存在
            os.makedirs(save_dir, exist_ok=True)
            best_model_path = os.path.join(save_dir, f"best_model_{dataset_name}.pt")
            torch.save(model.state_dict(), best_model_path)
            
            # 保存对应的tokenizer
            tokenizer_save_path = os.path.join(save_dir, f"tokenizer_{dataset_name}")
            tokenizer.save_pretrained(tokenizer_save_path)
            print(f"New best model saved at: {best_model_path}")
            print(f"Tokenizer saved at: {tokenizer_save_path}")
    
    print(f"Training completed. Best accuracy: {best_accuracy:.4f}")
    return best_model_path