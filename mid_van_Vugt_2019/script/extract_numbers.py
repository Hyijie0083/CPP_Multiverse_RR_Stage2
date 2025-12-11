import re

def extract_numbers(s):
    # 使用正则表达式提取所有数字（支持 . - _ 等分隔符）
    numbers = re.findall(r'\d+', s)
    # 转换为浮点数
    return [float(x) for x in numbers]
