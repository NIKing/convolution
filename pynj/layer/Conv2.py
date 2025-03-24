from .Module import Module

class Conv2(Module):
    def __init__(self, kernel_size=3, padding=(0, 0), stride=1):
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        # 初始化核矩阵
        self.kernel = np.random.randn(kernel_size, kernel_size) * np.sqrt(2. / kernel_size)

    def forward(self, input_ids):
        
        # 填充矩阵
        if self.padding != (0, 0):
            row_size, col_size = self.padding
            net_input_ids = np.pad(input_ids, pad_width=((row_size, row_size), (col_size, col_size)), mode='constant', constant_values=0)
        
        # 
        
