import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


img = cv.imread("corgi working.jpg")
cv.imshow('Original',img)

hsv_img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
print(img.shape[:])
cv.imshow('HSV',hsv_img)

def rgb_to_hsv(img):
  sample = (img * (1/255.0))
  blue,green,red = cv.split(sample)
  row,col,channel = sample.shape

  v = np.zeros(sample.shape[:2],dtype=np.float64)
  s = np.zeros(sample.shape[:2],dtype=np.float64)
  h = np.zeros(sample.shape[:2],dtype=np.float64)

  for i in range(row):
      for j in range(col):
          v[i,j] = max(blue[i,j],green[i,j],red[i,j])
          minimum = min(blue[i,j],green[i,j],red[i,j])

          if v[i,j] != 0.0:
              s[i,j] = ((v[i,j] - minimum) / v[i,j])
          else:
              s[i,j] = 0.0

          if v[i,j] == red[i,j]:
              h[i,j] = 60*(green[i,j] - blue[i,j])/(v[i,j] - minimum)
          elif v[i,j] == green[i,j]:
              h[i,j] = 120 + 60*(blue[i,j] - red[i,j])/(v[i,j] - minimum)
          elif v[i,j] == blue[i,j]:
              h[i,j] = 240 + 60*(red[i,j] - green[i,j])/(v[i,j] - minimum)

          if h[i,j] < 0:
              h[i,j] = h[i,j] + 360


  v = v*255.0
  s = s*255.0
  h = h/2
  hsv = np.round(cv.merge((v,s,h)))
  return hsv.astype(np.int) 

hsv_img_manual = rgb_to_hsv(img)
plt.imshow(hsv_img_manual)
plt.title('Manual Conversion of RGB to HSV Image')
plt.show()


