import pandas as pd
import re
import ast

def contains_ip_or_node(parameter_list):
    # IP地址的正则表达式模式（包括带端口号和可能的路径形式）
    ip_pattern = r'(?:\/|^|[^\d])(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?=\/|$|[^\d])'
    
    try:
        # 如果是字符串形式的列表，尝试转换
        if isinstance(parameter_list, str):
            # 首先直接在原始字符串中查找
            if re.search(ip_pattern, parameter_list):
                return True
                
            try:
                # 尝试解析为Python列表
                params = ast.literal_eval(parameter_list)
                if isinstance(params, (list, tuple)):
                    # 检查列表中的每个参数
                    for param in params:
                        if isinstance(param, str) and re.search(ip_pattern, param):
                            return True
            except:
                pass
        return False
    except:
        return False

def analyze_log_file(file_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 确保ParameterList列存在
        if 'ParameterList' not in df.columns:
            print("错误：文件中没有找到ParameterList列")
            return
        
        # 应用检查函数到每一行
        df['contains_ip'] = df['ParameterList'].apply(contains_ip_or_node)
        
        # 计算统计信息
        total_logs = len(df)
        logs_with_ip = df['contains_ip'].sum()
        percentage = (logs_with_ip / total_logs) * 100 if total_logs > 0 else 0
        
        print(f"总日志数: {total_logs}")
        print(f"包含IP或节点信息的日志数: {logs_with_ip}")
        print(f"占比: {percentage:.2f}%")
        
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")

if __name__ == "__main__":
    file_path = "HDFS.log_structured_new.csv"
    analyze_log_file(file_path) 