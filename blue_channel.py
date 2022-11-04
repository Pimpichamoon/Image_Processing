import cv2 as cv
import numpy as np
img = cv.imread('C:\\Users\\ASUS\\Desktop\\img pro\\corgi working.jpg')
cv.imshow('Pim',img)
print(img.shape[:])
blue = img[:,:,0]
cv.imshow('Blue',blue)


green = img[:,:,1]
cv.imshow('Green',green)


red = img[:,:,2]
cv.imshow('Red',red)
cv.waitKey(0)