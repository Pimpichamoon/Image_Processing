from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

img = cv.imread("corgi working.jpg")
cv.imshow('Original',img)

def separate_channels(img,channel):
    lab_image = np.zeros(img.shape)
    lab_image[:,:,channel] = img[:,:,channel]
    lab_image = lab2rgb(lab_image)
    return (lab_image)

lab = rgb2lab(img/255.0)
lab_L = separate_channels(lab,0)
lab_a = separate_channels(lab,1)
lab_b = separate_channels(lab,2)

plt.subplot(3,3,1)
plt.imshow(lab_L)
plt.title('Lab L')
plt.subplot(3,3,2)
plt.imshow(lab_a)
plt.title('Lab a')
plt.subplot(3,3,3)
plt.imshow(lab_b)
plt.title('Lab b')
plt.show()
