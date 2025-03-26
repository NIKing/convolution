import numpy as np

"""损失函数-S型函数"""
class Sigmoid():
    def __init__(self):
        self.name = 'ReLU'

        self.input = 0
        self.net_input = 0

        self.gradient = None

    def __call__(self, net_input):
        return self.forward(net_input)
            
    def forward(self, net_input):
        return 1 / (1 + np.exp(-net_input))

    def backward(self, net_input):
        return self.excute_fn(net_input) * (1 - self.execute_fn(net_input))

