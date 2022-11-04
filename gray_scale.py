from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

img = cv.imread("corgi working.jpg")
cv.imshow('Original',img)

gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
print(img.shape[:])
cv.imshow('Grayscale Image by opencv',gray_img)

def rgb_to_gray(img):
    gray_img_manual = np.zeros(img.shape)
    red = np.array(img[:,:,2])
    green = np.array(img[:,:,1])
    blue = np.array(img[:,:,0])
    avg = ((red*0.299)+(green*0.587)+(blue*0.114))
    gray_img_manual = img.copy()

    for i in range(3):
        gray_img_manual[:,:,i] = avg
    return gray_img_manual
    
gray_img_manual = rgb_to_gray(img)
plt.imshow(gray_img_manual)
plt.show()