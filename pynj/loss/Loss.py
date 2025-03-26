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
        
        # 计算【输出层】误差
        layer_error = layer_items[-1].backward()

        # 从后向前计算梯度
        for i in range(len(layer_items) - 1, -1, -1):
            inputs, net_input, weight, delta_fn = layer_items[i]
            
            #print(f'第{i}层误差', layer_error)

            # 获取上一层的输出结果, 若到了第一层，直接取输入值
            if i == 0:
                current_input = self.model.in_features
            else:
                current_input = inputs
            
            #print(f'第{i}层输入T', current_input.T)
            #layer_items[i]['gradient'] = np.dot(current_input.T, layer_error) / self.batch_size
            layer_item.backward(layer_error)

            # 记录当前误差, 计算【一般网络层】误差
            layer_error = delta_fn(net_input) * np.dot(layer_error, next_layer_weight.T)

            #print('*'*80)

        
                
