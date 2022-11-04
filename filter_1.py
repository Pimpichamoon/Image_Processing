import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
from scipy.signal import convolve2d
import cv2 as cv

img = cv.imread("corgi working.jpg")
img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)

blue_scaled = rescale(img1[:,:,0], 0.5)
green_scaled = rescale(img1[:,:,1], 0.5)
red_scaled = rescale(img1[:,:,2], 0.5)
img_scaled = np.stack([blue_scaled, green_scaled, red_scaled], axis=2)

Rows = int(input('Number of rows:'))  
Columns = int(input('Number of columns:'))  
   
matrix = []  
for i in range(Rows):    
    single_row = list(map(float, input().split()))  
    matrix.append(single_row)  
 
weight = sum(sum(matrix,[]))
kernel = (1/float(weight)) * np.array(matrix)
print(kernel)

def convolution(image, kernel):
    blue = convolve2d(image[:,:,0], kernel, 'valid')
    green = convolve2d(image[:,:,1], kernel, 'valid')
    red = convolve2d(image[:,:,2], kernel, 'valid')
    return np.stack([blue, green, red], axis=2)

conv_img = convolution(img_scaled, kernel).clip(0,1)
plt.subplot(121),plt.imshow(img1)
plt.subplot(122),plt.imshow(abs(conv_img))
plt.show()

