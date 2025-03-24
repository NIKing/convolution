import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 打印全部矩阵
np.set_printoptions(edgeitems=1000, linewidth=1000)

# 载入图片 
img = mpimg.imread('IMG_0146.JPG')
#img = mpimg.imread('222.jpg')

# 放大显示
#plt.figure(figsize=(10, 10))  # 可以调整图像显示的大小
#plt.imshow(img)
#plt.axis('off')  # 隐藏坐标轴
#plt.show()

#print(img.shape)
#print(img)

# 如果是RGB图，转换为灰度
if len(img.shape) == 3:
    img = np.mean(img, axis=2).astype(np.uint8)

crop = img
print(crop.shape)
#print(crop)
print()

plt.subplot(1, 2, 2)
plt.imshow(crop, cmap='gray', vmin=0, vmax=255)
plt.title("Zoomed Region with Values")

# 在局部区域上标注数值
for i in range(crop.shape[0]):
    for j in range(crop.shape[1]):
        color = 'white' if crop[i, j] < 128 else 'black'
        plt.text(j, i, str(crop[i, j]),
                 ha='center', va='center',
                 color=color, fontsize=10)

plt.figure(figsize=(10, 10))  
plt.axis('off')
plt.show()


