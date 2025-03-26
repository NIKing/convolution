import numpy as np

class Sigmoid():
    def __init__(self):
        self.name = 'ReLU'

    def __call__(self, net_input):
        """损失函数-S型函数"""
        return self.excute_fn(net_input)
            
    def execute_fn(self, net_input):
        return 1 / (1 + np.exp(-net_input))

    def delta(self, net_input):
        return self.excute_fn(net_input) * (1 - self.execute_fn(net_input))

