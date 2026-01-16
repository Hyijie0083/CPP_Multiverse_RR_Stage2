# 用递归函数解开各种嵌套
import numpy as np

def extract_scalar(x, default=np.nan):
    # 如果输入是空列表或空数组，返回默认值
    if isinstance(x, (list, np.ndarray)) and len(x) == 0:
        return default
    # 递归解开嵌套
    while isinstance(x, (list, np.ndarray)):
        # 检查是否为空
        if len(x) == 0:
            return default
        x = x[0]
    return x