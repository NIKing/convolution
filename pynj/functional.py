import numpy as np

def leaky_relu(x, alpha=0.01):
    """Leaky-Relu激活函数"""
    return np.where(x > 0, x, x * alpha)

def identical(net_input):
    """恒等激活函数"""
    return net_input

def soft_max(net_input):
    """软最大化函数 - 转概率分布"""
    return np.exp(net_input) / np.sum(np.exp(net_input), axis=-1, keepdims=True)

def soft_max_delta(net_input):
    pass

def arg_max(logits, axis=0):
    """取最大值"""
    return np.argmax(logits, axis) 
