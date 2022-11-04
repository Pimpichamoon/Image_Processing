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

#Plot histogram between opencv and manual
plt.subplot(2,2,1)
hist,bin = np.histogram(gray_img.ravel(),256,[0,255])
plt.xlim([0,255])
plt.plot(hist)
plt.title('Histogram of Grayscale Image by opencv')

plt.subplot(2,2,2)
hist,bin = np.histogram(gray_img_manual.ravel(),256,[0,255])
plt.xlim([0,255])
plt.plot(hist)
plt.title('Histogram of Grayscale Image by manual')
plt.show()

#Manual histogram
row1, col1 = gray_img.shape[0],gray_img.shape[1]
pixel1 = np.zeros((256),np.uint64)

for i in range(0,row1):
    for j in range(0,col1):
        pixel1[gray_img[i,j]] += 1 ## y-axis

x = np.arange(0,256)
plt.subplot(2,2,1)
plt.title('Manual Histogram of Grayscale Image by opencv')
plt.plot(x,pixel1)

row2, col2 = gray_img_manual.shape[0],gray_img_manual.shape[1]
pixel2 = np.zeros((256),np.uint64)

for i in range(0,row2):
    for j in range(0,col2):
        pixel2[gray_img_manual[i,j]] += 1 ## y-axis
        

x = np.arange(0,256)
plt.subplot(2,2,2)
plt.title('Manual Histogram of Grayscale Image by manual')
plt.plot(x,pixel2)
plt.show()
cv.waitKey(0)




