from pynj.layer import Module, Conv2, Linear

import pynj.functional as Fn

class Model(Module):
    def __init__(self, label_size=0):
        super(Model, self).__init__()

        self.conv = Conv2(kernel_size=3, padding=(1, 1), stride=1)
        self.linear = Linear(input_features=28, output_features=label_size)

    def forward(self, input_ids, is_train=False):
        conv_features = self.conv(input_ids)
        print(logist)

        logist = self.linear(conv_features)

        if is_train:
            return logist

        return Fn.soft_max(logist).argMax(dim=-1)
