import numpy as np

"""ReLU激活函数，值域为max(0, z)"""
class ReLU():
    def __init__(self):
        self.name = 'ReLU'
        
        self.input = 0
        self.net_input = 0

        self.gradient = None

    def __call__(self, net_input):
        return self.forward(net_input)

    def forward(self, net_input):
        self.input = net_input

        self.net_input = np.maximum(net_input, 0)

        return self.net_input

    def backward(self, net_input):
        return np.where(net_input, 1, 0)

