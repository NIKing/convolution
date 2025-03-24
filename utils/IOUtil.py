import numpy as np
 
def read_image_file(file_path):
    """加载 MNIST 图片数据"""
    with open(file_path, 'rb') as f:
        # 读取头部信息
        magic = int.from_bytes(f.read(4), byteorder='big')
        num_images = int.from_bytes(f.read(4), byteorder='big')
        rows = int.from_bytes(f.read(4), byteorder='big')
        cols = int.from_bytes(f.read(4), byteorder='big')
        
        # 读取剩余的像素数据
        images = np.frombuffer(f.read(), dtype=np.uint8)
        # 重构成 (num_images, rows, cols)
        images = images.reshape((num_images, rows, cols))

    return images
 
def read_label_file(file_path):
    """加载 MNIST 标签数据"""
    with open(file_path, 'rb') as f:
        # 读取头部信息
        magic = int.from_bytes(f.read(4), byteorder='big')
        num_labels = int.from_bytes(f.read(4), byteorder='big')
        
        # 读取标签数据
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels

