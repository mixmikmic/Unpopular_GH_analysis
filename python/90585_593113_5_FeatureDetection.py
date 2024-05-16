import cv2
import numpy as np
import matplotlib.pyplot as plt

edge = cv2.imread("../resources/harris_1.png", cv2.IMREAD_GRAYSCALE)
edge_x = cv2.Sobel(edge, cv2.CV_64F, 1, 0, ksize=5)
edge_y = cv2.Sobel(edge, cv2.CV_64F, 0, 1, ksize=5)

flat = cv2.imread("../resources/harris_2.png", cv2.IMREAD_GRAYSCALE)
flat_x = cv2.Sobel(flat, cv2.CV_64F, 1, 0, ksize=5)
flat_y = cv2.Sobel(flat, cv2.CV_64F, 0, 1, ksize=5)

corner = cv2.imread("../resources/harris_3.png", cv2.IMREAD_GRAYSCALE)
corner_x = cv2.Sobel(corner, cv2.CV_64F, 1, 0, ksize=5)
corner_y = cv2.Sobel(corner, cv2.CV_64F, 0, 1, ksize=5)

plt.subplot(3, 3, 1)
plt.imshow(edge, cmap='gray')
plt.subplot(3, 3, 2)
plt.imshow(edge_x, cmap='gray')
plt.subplot(3, 3, 3)
plt.imshow(edge_y, cmap='gray')

plt.subplot(3, 3, 4)
plt.imshow(flat, cmap='gray')
plt.subplot(3, 3, 5)
plt.imshow(flat_x, cmap='gray')
plt.subplot(3, 3, 6)
plt.imshow(flat_y, cmap='gray')

plt.subplot(3, 3, 7)
plt.imshow(corner, cmap='gray')
plt.subplot(3, 3, 8)
plt.imshow(corner_x, cmap='gray')
plt.subplot(3, 3, 9)
plt.imshow(corner_y, cmap='gray')

plt.show()

fig = plt.figure()
fig.set_size_inches(18, 5)
fig.add_subplot(1, 3, 1)
plt.scatter(edge_x.flatten(), edge_y.flatten())
fig.add_subplot(1, 3, 2)
plt.scatter(flat_x.flatten(), flat_y.flatten())
fig.add_subplot(1, 3, 3)
plt.scatter(corner_x.flatten(), corner_y.flatten())
plt.show()

image = cv2.imread("../resources/corner_test.jpg")
copy = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fig = plt.figure()
fig.set_size_inches(10, 5)
fig.add_subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

corner_regions = cv2.cornerHarris(gray, 3, 3, 0.04)
thresholded_region = corner_regions > 0.01 * corner_regions.max()
# print "Corners: ", image[thresholded_region]
image[thresholded_region] = [0, 255, 0]

fig.add_subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

# Refined Corners
# Use of cv2.cornerSubPixel

corner_regions = cv2.dilate(corner_regions, None)
refined_region = cv2.threshold(corner_regions, 0.001 * corner_regions.max(), 255, cv2.THRESH_BINARY)[1]
refined_region = np.uint8(refined_region)

ret, labels, stats, centroids = cv2.connectedComponentsWithStats(refined_region)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000000, 0.0001)
corners = cv2.cornerSubPix(gray, np.float32(centroids), (3, 3), (-1, -1), criteria)

res = np.int0(np.hstack((centroids, corners)))
copy[res[:, 1], res[:, 0]] = [255, 0, 0]
copy[res[:, 3], res[:, 2]] = [0, 255, 0]

plt.imshow(cv2.cvtColor(copy, cv2.COLOR_BGR2RGB))
plt.show()

image = cv2.imread("../resources/corner_test.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.goodFeaturesToTrack(image, number of corners, quality, min euclidean distance)
tomasi_corners = np.int0(cv2.goodFeaturesToTrack(gray, 20, 0.05, 8))
for corner in tomasi_corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 4, (0, 255, 0), 2)

fig = plt.figure()
fig.set_size_inches(5, 10)
fig.add_subplot(1, 1, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()



