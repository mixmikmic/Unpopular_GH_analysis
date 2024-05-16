import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('../resources/messi.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]
thresh2 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)[1]
thresh3 = cv2.threshold(gray, 80, 255, cv2.THRESH_TRUNC)[1]
thresh4 = cv2.threshold(gray, 80, 255, cv2.THRESH_TOZERO)[1]
thresh5 = cv2.threshold(gray, 80, 255, cv2.THRESH_TOZERO_INV)[1]

titles = ['GRAY', 'TRUNC', 'BINARY', 'BINARY_INV', 'TOZERO', 'TOZERO_INV']
plots = [gray, thresh3, thresh1, thresh2, thresh4, thresh5]

fig = plt.figure()
fig.set_size_inches(30, 30)

for i in range(6):
    fig.add_subplot(3, 2, i+1)
    plt.imshow(plots[i], cmap='gray')
    plt.title(titles[i])

plt.show()

mean_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 0)
gaussian_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 0)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(mean_thresh, cmap='gray')

fig.add_subplot(1, 2, 2)
plt.imshow(gaussian_thresh, cmap='gray')

plt.show()

thresh_selected, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print "Calculated threshold: %f" % (thresh_selected)

plt.imshow(otsu_thresh, cmap='gray')
plt.show()

