import numpy as np
import pynj.functional as Fn

from .Loss import Loss

class CrossEntropyLoss(Loss):
    def __init__(self, model = None):
        super(CrossEntropyLoss, self).__init__(model)

        self.loss = 0.0
        self.loss_error = [0.0]
        self.batch_size = 0

    def __call__(self, predict, target):
        # print('计算损失值')
        # https://www.jb51.net/article/274051.htm

        #print(Fn.SoftMax(predict))
        #print(target)
        #print(target * predict) 
        #print()
        
        self.batch_size = target.shape[0]

        # 预测概率分布
        logist = Fn.soft_max(predict)

        # 损失值，只是衡量模型“误差”
        self.loss = -np.mean(np.sum(target * np.log(logist), axis=1))
        
        # 输出层误差，它是输出层的梯度
        self.loss_error = logist - target

        return self
