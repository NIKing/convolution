from pynj.layer import Module, Conv2

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv = Conv2()

    def forward(self, input_ids, is_train=False):
        logist = self.conv(input_ids)
        print(logist)
