import cv2 as cv
import numpy as np
  
img = cv.imread("road.jpg") 
upper_threshold = 200  
lower_threshold = 25  
canny_img = cv.Canny(img, lower_threshold, upper_threshold)

new_img = img.copy()
img_gray = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
img_edges = cv.Canny(img_gray, 50, 150, apertureSize=3)
img_lines = cv.HoughLines(img_edges, 1, np.pi/180, 200)
for line in img_lines:
    rho,theta = line[0]

    theta_cos = np.cos(theta)
    theta_sin = np.sin(theta)

    x0 = theta_cos*rho
    y0 = theta_sin*rho

    x1 = int(x0 + 1000*(-theta_sin))
    y1 = int(y0 + 1000*(theta_cos))
    x2 = int(x0 - 1000*(-theta_sin))
    y2 = int(y0 - 1000*(theta_cos))

    cv.line(new_img,(x1,y1),(x2,y2),(0,0,255),2)
  
cv.imshow('Original', img)
cv.imshow('Canny Operator', canny_img)
cv.waitKey(0)

cv.imshow('Original (Hough Transform)', new_img)
cv.imwrite('Hough Transform.jpg',img_gray)
hough_img = cv.imread('Hough Transform.jpg') 
cv.imshow('Hough Transform', hough_img)
cv.waitKey(0)



