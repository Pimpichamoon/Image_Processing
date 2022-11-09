import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.transform import rescale
import numpy as np
import cv2 as cv

img1 = cv.imread("Lady.jpg")

blue_scaled = rescale(img1[:,:,0], 0.5)
green_scaled = rescale(img1[:,:,1], 0.5)
red_scaled = rescale(img1[:,:,2], 0.5)
img_scaled = np.stack([blue_scaled, green_scaled, red_scaled], axis=2)

def compass_operator(image):
    Hs = np.array([[[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]],

            [[-2, -1, 0],[-1, 0, 1],[0, 1, 2]],

            [[-1, -2, -1],[0, 0, 0],[1, 2, 1]],

            [[0, -1, -2],[1, 0, -1],[2, 1, 0]],

            [[1, 0, -1],[2, 0, -2],[1, 0, -1]],

            [[2, 1, 0],[1, 0, -1],[0, -1, -2]],

            [[1, 2, 1],[0, 0, 0],[-1, -2, -1]],

            [[0, 1, 2],[-1, 0, 1],[-2, -1, 0]]])
    
    for i in range(8):
        blue = convolve2d(image[:,:,0], Hs[i], 'valid')
        green = convolve2d(image[:,:,1], Hs[i], 'valid')
        red = convolve2d(image[:,:,2], Hs[i], 'valid')
    return np.stack([blue, green, red], axis=2)

def convolution(image,kernel):
    blue = convolve2d(image[:,:,0], kernel, 'valid')
    green = convolve2d(image[:,:,1], kernel, 'valid')
    red = convolve2d(image[:,:,2], kernel, 'valid')
    return np.stack([blue, green, red], axis=2)

edge_sharpening = np.array([[1,1,1],[1,-8,1],[1,1,1]])
edge_sharpening_output_img = convolution(img_scaled, edge_sharpening).clip(0,1)

unsharp_mask = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
unsharp_mask_output_img = convolution(img_scaled, unsharp_mask).clip(0,1)

compass_output_img = compass_operator(img_scaled).clip(0,1)

plt.subplot(221),plt.imshow(img1)
plt.title('Original')
plt.subplot(222),plt.imshow(abs(compass_output_img))
plt.title('Compass Filter')
plt.subplot(223),plt.imshow(abs(edge_sharpening_output_img))
plt.title('Edge Sharpening')
plt.subplot(224),plt.imshow(abs(unsharp_mask_output_img))
plt.title('Unsharp Mask')
plt.show()


