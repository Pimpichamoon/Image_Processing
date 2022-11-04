from matplotlib.cbook import flatten
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

img = cv.imread("corgi working.jpg")
img_1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_2 = cv.imread("Lady.jpg")
M_filter = 3
N_filter = 3

## Max Filetr ##
def max_filter(image):
    image_size = image.shape
    h_image = image_size[0]
    w_image = image_size[1]
    max_filter = np.zeros([(h_image - M_filter), (w_image - N_filter)])
    for i in range(max_filter.shape[0]):
        for j in range(max_filter.shape[1]):
            max_filter[i,j] = max((image[i:i+M_filter, j:j+N_filter]).flatten())
    return max_filter.astype(np.uint16)

#max_img_b = max_filter(img_1[:,:,0])
#max_img_g = max_filter(img_1[:,:,1])
#max_img_r = max_filter(img_1[:,:,2])
#max_img_1_1 =  cv.merge([max_img_b,max_img_g,max_img_r])
max_img_1 = max_filter(img_1)
max_img_2 = max_filter(img_2)
max_img_2_cvt = cv.cvtColor(max_img_2, cv.COLOR_RGB2BGR)

plt.figure(1)
plt.subplot(221),plt.imshow(img)
plt.subplot(222),plt.imshow(img_2)
plt.subplot(223),plt.imshow(max_img_1)
plt.subplot(224),plt.imshow(max_img_2)
plt.show()

## Min Filter ##
def min_filter(image):
    image_size = image.shape
    h_image = image_size[0]
    w_image = image_size[1]
    min_filter = np.zeros([(h_image - M_filter), (w_image - N_filter)])
    for i in range(min_filter.shape[0]):
        for j in range(min_filter.shape[1]):
            min_filter[i,j] = min((image[i:i+M_filter, j:j+N_filter]).flatten())
    return min_filter.astype(np.uint16)

min_img_b = min_filter(img_1[:,:,0])
min_img_g = min_filter(img_1[:,:,1])
min_img_r = min_filter(img_1[:,:,2])
min_img_1 =  cv.merge([min_img_b,min_img_g,min_img_r])

min_img_2 = min_filter(img_2)
min_img_2_cvt = cv.cvtColor(min_img_2, cv.COLOR_RGB2BGR)

plt.figure(2)
plt.subplot(221),plt.imshow(img_1)
plt.subplot(222),plt.imshow(img_2)
plt.subplot(223),plt.imshow(min_img_1)
plt.subplot(224),plt.imshow(min_img_2_cvt)
plt.show()

## Median Filetr ##
def median_filter(image):
    image_size = image.shape
    h_image = image_size[0]
    w_image = image_size[1]
    median_filter = np.zeros([(h_image - M_filter), (w_image - N_filter)])
    for i in range(median_filter.shape[0]):
        for j in range(median_filter.shape[1]):
            median_filter[i,j] = np.median((image[i:i+M_filter, j:j+N_filter]).flatten())
    return median_filter.astype(np.uint16)

median_img_b = median_filter(img_1[:,:,0])
median_img_g = median_filter(img_1[:,:,1])
median_img_r = median_filter(img_1[:,:,2])
median_img_1 =  cv.merge([median_img_b,median_img_g,median_img_r])

median_img_2 = median_filter(img_2)
median_img_2_cvt = cv.cvtColor(median_img_2, cv.COLOR_RGB2BGR)

plt.figure(3)
plt.subplot(221),plt.imshow(img_1)
plt.subplot(222),plt.imshow(img_2)
plt.subplot(223),plt.imshow(median_img_1)
plt.subplot(224),plt.imshow(median_img_2_cvt)
plt.show()

## Weighted Median Filter ##
def weighted_median_filter(image, weighted):
    image_size = image.shape
    h_image = image_size[0]
    w_image = image_size[1]

    weighted_median_filter = np.zeros([(h_image - M_filter), (w_image - N_filter)])
    W = weighted.flatten()

    for i in range(weighted_median_filter.shape[0]):
        for j in range(weighted_median_filter.shape[1]):
            array_image = (image[i:i+M_filter, j:j+N_filter]).flatten()
            
            array = []
            for k in range(len(array_image)):
                array_app = array.append(np.repeat(array_image[k], W[k]))

            array_app = flatten(array_app)
            weighted_median_filter[i,j] = np.median(np.array(array_app))
    return weighted_median_filter.astype(np.uint8)

weighted = np.array(([1,2,1],[2,3,2],[1,2,1]))
weighted_median_img_b = weighted_median_filter(img_1[:,:,0], weighted)
weighted_median_img_g = weighted_median_filter(img_1[:,:,1], weighted)
weighted_median_img_r = weighted_median_filter(img_1[:,:,2], weighted)
weighted_median_img_1 =  cv.merge([weighted_median_img_b,weighted_median_img_g,weighted_median_img_r])
weighted_median_img_2 = weighted_median_filter(img_2, weighted)
weighted_median_img_2_cvt = cv.cvtColor(weighted_median_img_2, cv.COLOR_RGB2BGR)

plt.figure(4)
plt.subplot(221),plt.imshow(img_1)
plt.subplot(222),plt.imshow(img_2)
plt.subplot(223),plt.imshow(weighted_median_img_1)
plt.subplot(224),plt.imshow(weighted_median_img_2_cvt)
plt.show()


