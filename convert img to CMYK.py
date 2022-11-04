import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


bgr_img = cv.imread("corgi working.jpg")
cv.imshow('Original',bgr_img)

bgr_img_float = (1/255.0) * (bgr_img.astype(np.float)) 

K = 1 - np.max(bgr_img_float, axis=2)
C = (1-bgr_img_float[:,:,2]-K)/(1-K)
cv.imshow('C',C)

M = (1-bgr_img_float[:,:,1]-K)/(1-K)
cv.imshow('M',M)

Y = (1-bgr_img_float[:,:,0]-K)/(1-K)
cv.imshow('Y',Y)

CMYK_img = (np.dstack((C,M,Y,K)) * 255).astype(np.uint8) 

plt.imshow(CMYK_img)
plt.show()

