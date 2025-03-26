from pynj.layer import Module, Conv2, Linear

import pynj.functional as Fn

class Model(Module):
    def __init__(self, label_size=0):
        super(Model, self).__init__()

        self.conv = Conv2(kernel_size=(3,3), padding=(0,0), stride=1)
        self.linear = Linear(input_features=26 * 26, output_features=label_size)

        self.layers = [self.conv, self.linear]

    def forward(self, input_ids, is_train=True):

        # 卷积
        conv_features = self.conv(input_ids)
        conv_features = Fn.relu(conv_features)
        
        # 线性分类
        b, i, j = conv_features.shape
        sequence = conv_features.reshape(b, i * j)
        
        logist = self.linear(sequence)

        if is_train:
            return logist

        return Fn.arg_max(Fn.soft_max(logist), axis=-1)
