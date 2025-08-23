
import pandas as pd
import numpy as np

def smooth(x, epoch_time, sample_rate):
    # 确保 x 是 DataFrame
    x2 = x.copy()
    
    # 计算窗口大小（0.2*sample_rate，因为窗口是从 i-0.1*sample_rate 到 i+0.1*sample_rate）
    window_size = int(0.2 * sample_rate) + 1  # 确保窗口大小是奇数，包含中心点
    
    # 使用 rolling 方法计算滑动窗口均值
    x2 = x2.rolling(window=window_size, center=True, min_periods=1).mean()
    
    # 对于边界（前0.1*sample_rate和后0.1*sample_rate），rolling 会自动用较小的窗口填充
    return x2
