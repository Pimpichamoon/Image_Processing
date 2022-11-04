import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('Lady.jpg',0)

array_1 = np.zeros((img.shape[0],img.shape[1]), dtype="uint8")

aq_low = np.quantile(img,0.01)
aq_high = np.quantile(img,1-0.01)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j]  <= aq_low:
            array_1[i,j] = 0
        if aq_low < img[i,j] < aq_high:
            array_1[i,j] = 0 + (img[i,j] - aq_low)*((255)/(aq_high - aq_low))
        if img[i,j] >= aq_high:
            array_1[i,j] = 255

equ = cv.equalizeHist(array_1)

hist,bins = np.histogram(equ.flatten(),256,[0,255])
cdf = hist.cumsum()
cdf_normalized = cdf * ((hist.max()) / (equ.shape[0]*equ.shape[1]))
plt.plot(cdf_normalized, color = 'b')
plt.hist(equ.flatten(),256,[0,255], color = 'r')
plt.legend('cdf','histogram')
plt.show()
