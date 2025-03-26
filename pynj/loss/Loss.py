import numpy as np

class Loss():
    def __init__(self, model = None):
        self.model = model

        self.loss = 0.0
        self.loss_error = [0.0]
        self.batch_size = 0

    def __call__(self, predict, target):
        return self

    def item(self):
        return self.loss

    def backward(self):
        """反向传播的计算方式是通过损失函数/权重，得到梯度值"""
        next_layer_error = 0.0
        next_layer_weight = []
        
        layer_items = list(self.model.layers)
         
        # 从后向前计算梯度
        for i in range(len(layer_items) - 1, -1, -1):
            inputs, net_input, weight, delta_fn = layer_items[i]
            
            # 计算误差，这里误差指的既不是【上一层误差】也不是【当前层误差】，而是上一层与当前层连接的【变化率】
            if i == len(layer_items) - 1:
                # 计算【输出层】误差
                layer_error = self.loss_error
            else:
                # 计算【一般网络层】误差
                layer_error = delta_fn(net_input) * np.dot(next_layer_error, next_layer_weight.T)
            
            #print(f'第{i}层误差', layer_error)

            # 获取上一层的输出结果, 若到了第一层，直接取输入值
            if i == 0:
                current_input = self.model.in_features
            else:
                current_input = inputs
            
            #print(f'第{i}层输入T', current_input.T)

            # 记录当前信息，用于误差传播
            next_layer_error = layer_error
            next_layer_weight = weight 

            #print('*'*80)

        
                
