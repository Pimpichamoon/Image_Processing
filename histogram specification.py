import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms

img = cv.imread('Lady.jpg')
reference = cv.imread('corgi working.jpg')
matched = match_histograms(img, reference, multichannel=True)
cv.imshow('Source',img)
cv.imshow('Reference',reference)
cv.imshow('Matched',matched)

##Blue Channel##
plt.subplot(3,3,1)
img_hist, bins = exposure.histogram(img[..., 0], source_range='dtype')
plt.plot(bins, img_hist / img_hist.max())
img_cdf, bins = exposure.cumulative_distribution(img[..., 0])
plt.plot(bins, img_cdf)

plt.subplot(3,3,2)
reference_hist, bins = exposure.histogram(reference[..., 0], source_range='dtype')
plt.plot(bins, reference_hist / reference_hist.max())
reference_cdf, bins = exposure.cumulative_distribution(reference[..., 0])
plt.plot(bins, reference_cdf)

plt.subplot(3,3,3)
matched_hist, bins = exposure.histogram(matched[..., 0], source_range='dtype')
plt.plot(bins, matched_hist / matched_hist.max())
matched_cdf, bins = exposure.cumulative_distribution(matched[..., 0])
plt.plot(bins, matched_cdf)

##Green Channel##
plt.subplot(3,3,4)
img_hist, bins = exposure.histogram(img[..., 1], source_range='dtype')
plt.plot(bins, img_hist / img_hist.max())
img_cdf, bins = exposure.cumulative_distribution(img[..., 1])
plt.plot(bins, img_cdf)

plt.subplot(3,3,5)
reference_hist, bins = exposure.histogram(reference[..., 1], source_range='dtype')
plt.plot(bins, reference_hist / reference_hist.max())
reference_cdf, bins = exposure.cumulative_distribution(reference[..., 1])
plt.plot(bins, reference_cdf)

plt.subplot(3,3,6)
matched_hist, bins = exposure.histogram(matched[..., 1], source_range='dtype')
plt.plot(bins, matched_hist / matched_hist.max())
matched_cdf, bins = exposure.cumulative_distribution(matched[..., 1])
plt.plot(bins, matched_cdf)

##Red Channel##
plt.subplot(3,3,7)
img_hist, bins = exposure.histogram(img[..., 2], source_range='dtype')
plt.plot(bins, img_hist / img_hist.max())
img_cdf, bins = exposure.cumulative_distribution(img[..., 2])
plt.plot(bins, img_cdf) 

plt.subplot(3,3,8)
reference_hist, bins = exposure.histogram(reference[..., 2], source_range='dtype')
plt.plot(bins, reference_hist / reference_hist.max())
reference_cdf, bins = exposure.cumulative_distribution(reference[..., 2])
plt.plot(bins, reference_cdf)

plt.subplot(3,3,9)
matched_hist, bins = exposure.histogram(matched[..., 2], source_range='dtype')
plt.plot(bins, matched_hist / matched_hist.max())
matched_cdf, bins = exposure.cumulative_distribution(matched[..., 2])
plt.plot(bins, matched_cdf)
plt.show()




