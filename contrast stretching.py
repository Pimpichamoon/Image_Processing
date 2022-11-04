from matplotlib.image import imread
import matplotlib.pyplot as plt
import skimage
import numpy as np
import cv2 as cv

img = cv.imread("Lady.jpg",0)
cv.imshow('Original',img)

array_1 = np.zeros((img.shape[0],img.shape[1]), dtype="uint8")

a_low = np.min(img)
a_high = np.max(img)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        array_1[i,j] = (img[i,j] - a_low)*(255/(a_high - a_low))

cv.imshow('Contrast',array_1)


array_2 = np.zeros((img.shape[0],img.shape[1]), dtype="uint8")

aq_low = np.quantile(img,0.01)
aq_high = np.quantile(img,1-0.01)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j]  <= aq_low:
            array_2[i,j] = 0
        if aq_low < img[i,j] < aq_high:
            array_2[i,j] = 0 + (img[i,j] - aq_low)*((255)/(aq_high - aq_low))
        if img[i,j] >= aq_high:
            array_2[i,j] = 255

cv.imshow('Modified Contrast',array_2)

array_3 = np.zeros((img.shape[0],img.shape[1]), dtype="uint8")
threshold = skimage.filters.threshold_otsu(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j] > threshold:
            array_3[i,j] = 255
        else:
            array_3[i,j] = 0
            
cv.imshow('Thresholding',array_3)

plt.subplot(1,3,1)
hist,bin = np.histogram(img.ravel(),256,[0,255])
plt.xlim([0,255])
plt.plot(hist)
plt.title('Original')

plt.subplot(1,3,2)
hist,bin = np.histogram(array_1.ravel(),256,[0,255])
plt.xlim([0,255])
plt.plot(hist)
plt.title('Histogram of Auto-Contrast')

plt.subplot(1,3,3)
hist,bin = np.histogram(array_2.ravel(),256,[0,255])
plt.xlim([0,255])
plt.plot(hist)
plt.title('Histogram of Modified Auto-Contrast')
plt.show()



