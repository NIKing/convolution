import cv2

# 载入图片
image = cv2.imread('IMG_0146.JPG')

# 放大图片
image_resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# 显示图片
cv2.imshow('Resized Image', image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

