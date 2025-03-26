import numpy as np

class ReLU():
    def __init__(self):
        self.name = 'ReLU'
        self.input = 0
        self.net_input = 0

        self.gradient = None

    def __call__(self, net_input):
        self.input = net_input

        """ReLU激活函数，值域为max(0, z)"""
        self.net_input = np.maximum(net_input, 0)

        return self.net_input

    def delta(self, net_input):
        """ReLU导函数"""
        return np.where(net_input, 1, 0)


    def gradient(self, net_input):
        return self.delta(net_input)


