from PIL import Image
import numpy as np
import cv2
import os
a = [20,80,140,200]
img = cv2.imread(r'C:\Users\lenks\Desktop\2.jpg')
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for z in range(img.shape[2]):
            img[i][j][z] = a[img[i][j][z]%4]

print(img)
cv2.imwrite(r'C:\Users\lenks\Desktop\23.jpg',img)