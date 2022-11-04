from cv2 import magnitude
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.transform import rescale
from scipy import ndimage
import numpy as np
import cv2 as cv

img = cv.imread("corgi working.jpg")
img_1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_1_gray = cv.imread("corgi working.jpg",0).astype('float64')
img_2 = cv.imread("Lady.jpg")
img_2_gray = cv.imread("Lady_color.png",0).astype('float64')

def operation(img, operation_x, operation_y):
    operation_axis_x = convolve2d(img, operation_y)
    operation_axis_y = convolve2d(img, operation_y)
    operation = (abs(operation_axis_x)**2) + (abs(operation_axis_y)**2)
    output_image = (abs(np.sqrt(operation)))
    return output_image

###### Prewitt Operation #####
prewitt_operation_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewitt_operation_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

output_img_1 = operation(img_1_gray, prewitt_operation_x, prewitt_operation_y)
output_img_2 = operation(img_2_gray, prewitt_operation_x, prewitt_operation_y)
plt.figure(1)
plt.subplot(221),plt.imshow(img_1)
plt.subplot(222),plt.imshow(output_img_1)
plt.subplot(223),plt.imshow(img_2)
plt.subplot(224),plt.imshow(output_img_2)
plt.show()

##### Sobel Operation #####
sobel_operation_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_operation_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

output_img_1 = operation(img_1_gray, sobel_operation_x, sobel_operation_y)
output_img_2 = operation(img_2_gray, sobel_operation_x, sobel_operation_y)
plt.figure(2)
plt.subplot(221),plt.imshow(img_1)
plt.subplot(222),plt.imshow(output_img_1)
plt.subplot(223),plt.imshow(img_2)
plt.subplot(224),plt.imshow(output_img_2)
plt.show()

###### Robert Operation #####
def robert_operation(image, robert_operation_x, robert_operation_y):
    operation_axis_x = ndimage.convolve(image, robert_operation_x)
    operation_axis_y = ndimage.convolve(image, robert_operation_y)
    output_image = np.sqrt((operation_axis_x)**2 + (operation_axis_y)**2)
    return output_image

robert_operation_x = np.array([[1,0],[0,-1]])
robert_operation_y = np.array([[0,1],[-1,0]])

output_img_1 = robert_operation(img_1_gray, robert_operation_x, robert_operation_y)
output_img_2 = robert_operation(img_2_gray, robert_operation_x, robert_operation_y)
plt.figure(3)
plt.subplot(221),plt.imshow(img_1)
plt.subplot(222),plt.imshow(output_img_1)
plt.subplot(223),plt.imshow(img_2)
plt.subplot(224),plt.imshow(output_img_2)
plt.show()



