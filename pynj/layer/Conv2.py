import math
import numpy as np

class Conv2():
    def __init__(self, kernel_size=(3,3), padding=(0,0), stride=1):

        self.name = 'conv2'

        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        # 随机初始化核矩阵
        m, n = kernel_size 
        self.kernel = np.random.randn(m, n) * np.sqrt(2. / m)
    
    def __call__(self, input_ids):
        #print(input_ids.shape)

        # 获取输入矩阵大小以及核矩阵大小
        b, i, j = input_ids.shape
        m, n = self.kernel.shape

        # 获取填充大小
        p, q = self.padding

        # 填充矩阵, 注意input_ids是三维的，那么pad_width需要根据维度大小也同样设置，若某个维度不填充可设置(0,0)
        net_input_ids = np.pad(input_ids, pad_width=((0, 0), (p, p), (q, q)), mode='constant', constant_values=0)

        # 计算卷积的输出大小 
        k = self._calculate_output_conv_size(i, p, m)
        l = self._calculate_output_conv_size(j, q, n)

        s = self.stride
        #print('输出大小', k, l)

        # 滑动- 检测特征 - convolution
        net_input = np.zeros((b, k, l), dtype=np.int32)
        for _b in range(b):
            for _i in range(k):
                for _j in range(l):
                    children_matrix = net_input_ids[_b][(_i * s):(m + _i * s), (_j * s):(m + _j * s)]
                    net_input[_b, _i, _j] = np.sum(children_matrix * self.kernel)
        
        #print(net_input)
        #print(net_input.shape)
        #print('===='*20)
        return net_input
    
    def _calculate_output_conv_size(self, input_size, padding_size, kernel_size):
        """
        计算卷积输出大小
        -param input_size 输入矩阵大小
        -param padding_size 填充大小
        -param kernel_size 核矩阵大小
        """
        return math.floor((input_size + 2 * padding_size - kernel_size) / self.stride) + 1
        

        
