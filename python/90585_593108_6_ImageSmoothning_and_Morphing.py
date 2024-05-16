import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('../resources/lena.jpg')

blurred1 = cv2.blur(image, (5, 5))
blurred2 = cv2.boxFilter(image, -1, (5, 5))  # -1 means same depth as source image

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

fig.add_subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(blurred1, cv2.COLOR_BGR2RGB))

fig.add_subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(blurred2, cv2.COLOR_BGR2RGB))

plt.show()

gauss_blur = cv2.GaussianBlur(image, (9, 9), 3)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

fig.add_subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(gauss_blur, cv2.COLOR_BGR2RGB))

plt.show()

extra_noise = cv2.imread('../resources/noise.jpg')

median_blur = cv2.medianBlur(extra_noise, 5)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(extra_noise, cv2.COLOR_BGR2RGB))

fig.add_subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB))

plt.show()

bi_blur = cv2.bilateralFilter(image, 9, 120, 100)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(gauss_blur, cv2.COLOR_BGR2RGB))

fig.add_subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(bi_blur, cv2.COLOR_BGR2RGB))

plt.show()

# let's have a look at structuring elements first
# 3 major structure elements - rectangle, ellipse, cross

# Rectangular Kernel
print "rectangle"
print cv2.getStructuringElement(cv2.MORPH_RECT, (8, 10))
print
# Elliptical Kernel
print "ellipse"
print cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 10))
print
# Crossed Kernel
print "cross"
print cv2.getStructuringElement(cv2.MORPH_CROSS, (8, 10))
print

letter = cv2.imread('../resources/letter1.png', cv2.IMREAD_GRAYSCALE)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded = cv2.erode(letter, kernel, iterations=1)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(letter, cmap='gray')

fig.add_subplot(1, 2, 2)
plt.imshow(eroded, cmap='gray')

plt.show()

dilated = cv2.dilate(letter, kernel, iterations=1)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(letter, cmap='gray')

fig.add_subplot(1, 2, 2)
plt.imshow(dilated, cmap='gray')

plt.show()

opening = cv2.morphologyEx(letter, cv2.MORPH_OPEN, kernel)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(letter, cmap='gray')

fig.add_subplot(1, 2, 2)
plt.imshow(opening, cmap='gray')

plt.show()

letter = cv2.imread('../resources/letter2.png', cv2.IMREAD_GRAYSCALE)

closing = cv2.morphologyEx(letter, cv2.MORPH_CLOSE, kernel)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(letter, cmap='gray')

fig.add_subplot(1, 2, 2)
plt.imshow(closing, cmap='gray')

plt.show()

letter = cv2.imread('../resources/letter1.png', cv2.IMREAD_GRAYSCALE)

outline = cv2.morphologyEx(letter, cv2.MORPH_GRADIENT, kernel)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(letter, cmap='gray')

fig.add_subplot(1, 2, 2)
plt.imshow(outline, cmap='gray')

plt.show()

