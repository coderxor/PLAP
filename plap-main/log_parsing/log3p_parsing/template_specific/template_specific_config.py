import os
import torch

# 继承原有的通用配置
from log_parsing.log3p_parsing.config import *

# 模板特定参数的配置
TEMPLATE_SPECIFIC = {
    "enabled": True,  # 是否启用模板特定参数
    "param_prefix": "PARAM",  # 参数标记前缀
}

# 实验配置扩展
EXPERIMENT.update({
    "template_specific": True,  # 标记这是模板特定的实验
    "metrics": ["template_pa", "template_ga"]  # 添加模板特定的评估指标
}) 

