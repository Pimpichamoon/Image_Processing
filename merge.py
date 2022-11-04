import cv2 as cv
import numpy as np
img = cv.imread('C:\\Users\\ASUS\\Desktop\\img pro\\corgi working.jpg')
cv.imshow('Pim',img)
print(img.shape[:])
b,g,r = cv.split(img)
cv.imshow('Blue',b)
cv.imshow('Green',g)
cv.imshow('Red',r)
img_merge = cv.merge([b,g,r])
cv.imshow('RGB_Image',img_merge)
cv.waitKey(0)