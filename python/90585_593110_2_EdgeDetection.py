import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../resources/messi.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (3, 3), 0)

sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

# sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
sobel = np.sqrt(np.square(sobelx) + np.square(sobely))

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(2, 2, 1)
plt.imshow(gray, cmap='gray')

fig.add_subplot(2, 2, 2)
plt.imshow(sobelx, cmap='gray')

fig.add_subplot(2, 2, 3)
plt.imshow(sobely, cmap='gray')

fig.add_subplot(2, 2, 4)
plt.imshow(sobel, cmap='gray')

plt.show()

from scipy import ndimage

roberts_x = np.array([[1, 0], [0, -1]])
roberts_y = np.array([[0, 1], [-1, 0]])

img = np.asarray(blurred, dtype="int32")

vertical = ndimage.convolve(img, roberts_y)
horizontal = ndimage.convolve(img, roberts_x)

robert = np.sqrt(np.square(vertical) + np.square(horizontal))

plt.imshow(robert, cmap='gray')
plt.show()

prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

horizontal = ndimage.convolve(img, prewitt_x)
vertical = ndimage.convolve(img, prewitt_y)

prewitt = np.sqrt(np.square(horizontal) + np.square(vertical))

plt.imshow(prewitt, cmap='gray')
plt.show()

def rotate_45(array):
    result = np.zeros_like(array)
    for i in range(3):
        for j in range(3):
            if i == 0 and j == 0:   result[i+1][j] = array[i][j]
            elif i == 2 and j == 0: result[i][j+1] = array[i][j]
            elif i == 2 and j == 2: result[i-1][j] = array[i][j]
            elif i == 0 and j == 2: result[i][j-1] = array[i][j]
            elif i == 0:            result[i][j-1] = array[i][j]
            elif j == 0:            result[i+1][j] = array[i][j]
            elif i == 2:            result[i][j+1] = array[i][j]
            elif j == 2:            result[i-1][j] = array[i][j]
    return result

kernel_n = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
kernel_nw = rotate_45(kernel_n)
kernel_w = rotate_45(kernel_nw)
kernel_sw = rotate_45(kernel_w)
kernel_s = rotate_45(kernel_sw)
kernel_se = rotate_45(kernel_s)
kernel_e = rotate_45(kernel_se)
kernel_ne = rotate_45(kernel_e)

d1 = ndimage.convolve(img, kernel_n)
d2 = ndimage.convolve(img, kernel_nw)
d3 = ndimage.convolve(img, kernel_w)
d4 = ndimage.convolve(img, kernel_sw)
d5 = ndimage.convolve(img, kernel_s)
d6 = ndimage.convolve(img, kernel_se)
d7 = ndimage.convolve(img, kernel_e)
d8 = ndimage.convolve(img, kernel_ne)

kirsch = np.sqrt(np.square(d1) + np.square(d2) + np.square(d3) + np.square(d4) +  
                 np.square(d5) + np.square(d6) + np.square(d7) + np.square(d8))

fig = plt.figure()
fig.set_size_inches(36, 20)

fig.add_subplot(3, 4, 1)
plt.imshow(d1, cmap='gray')

fig.add_subplot(3, 4, 2)
plt.imshow(d2, cmap='gray')

fig.add_subplot(3, 4, 3)
plt.imshow(d3, cmap='gray')

fig.add_subplot(3, 4, 4)
plt.imshow(d4, cmap='gray')

fig.add_subplot(3, 4, 5)
plt.imshow(d5, cmap='gray')

fig.add_subplot(3, 4, 6)
plt.imshow(d6, cmap='gray')

fig.add_subplot(3, 4, 7)
plt.imshow(d7, cmap='gray')

fig.add_subplot(3, 4, 8)
plt.imshow(d8, cmap='gray')

fig.add_subplot(3, 4, 10)
plt.imshow(img, cmap='gray')

fig.add_subplot(3, 4, 11)
plt.imshow(kirsch, cmap='gray')

plt.show()

laplacian = cv2.Laplacian(blurred, cv2.CV_16U)

plt.imshow(laplacian, cmap='gray')
plt.show()

canny_inbuilt = cv2.Canny(gray, 100, 200)

fig = plt.figure()
fig.set_size_inches(18, 10)

fig.add_subplot(1, 2, 1)
plt.imshow(canny_inbuilt, cmap='gray')

def canny_experimental(image):
    # Noise Supression
    sigma = 1.4  # experimental
    blurred = cv2.GaussianBlur(image, (5, 5), sigma)
    blurred = np.asarray(blurred, dtype="int32")
    
    # Intensity Gradient
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobelx = ndimage.convolve(blurred, kernel_x)
    sobely = ndimage.convolve(blurred, kernel_y)
    sobel = np.hypot(sobelx, sobely)
    
    ## Approx directions to horizontal, vertical, left diagonal and right diagonal
    sobel_dir = np.arctan2(sobely, sobelx)
    for x in range(sobel_dir.shape[0]):
        for y in range(sobel_dir.shape[1]):
            dir = sobel_dir[x][y]
            if 0<= dir < 22.5 or 157.5 <= dir <= 202.5 or 337.5 <= dir <= 360:
                sobel_dir[x][y] = 0  # horizontal
            elif 22.5 <= dir < 67.5 or 202.5 <= dir < 247.5:
                sobel_dir[x][y] = 45  # left diagonal
            elif 67.5 <= dir < 112.5 or 247.5 <= dir < 292.5:
                sobel_dir[x][y] = 90  # vertical
            else:
                sobel_dir[x][y] = 135  # right diagonal
    
    # Non-maxima Supression
    sobel_magnitude = np.copy(sobel)
    for x in range(1, sobel.shape[0] - 1):
        for y in range(1, sobel.shape[1] - 1):
            # Compare magnitude of gradients in the direction of gradient of pixel
            # If value is less than any of the neighoring magnitude, make pixel 0
            # We are checking in 3x3 neighbourhood
            if sobel_dir[x][y] == 0 and sobel[x][y] <= min(sobel[x][y+1], sobel[x][y-1]):
                    sobel_magnitude[x][y] = 0
            elif sobel_dir[x][y] == 45 and sobel[x][y] <= min(sobel[x-1][y+1], sobel[x+1][y-1]):
                sobel_magnitude[x][y] = 0
            elif sobel_dir[x][y] == 90 and sobel[x][y] <= min(sobel[x-1][y], sobel[x+1][y]):
                sobel_magnitude[x][y] = 0
            elif sobel[x][y] <= min(sobel[x+1][y+1], sobel[x-1][y-1]):
                sobel_magnitude[x][y] = 0
    
    # Double Thresholding
    sobel = sobel_magnitude
    canny = np.zeros_like(sobel)
    strong_edge = np.zeros_like(sobel)
    weak_edge = np.zeros_like(sobel)
    thresh = np.max(sobel)
    maxThresh = 0.2 * thresh
    minThresh = 0.1 * thresh
    for i in range(sobel.shape[0]):
        for j in range(sobel.shape[1]):
            if sobel[i][j] >= maxThresh:
                canny[i][j] = sobel[i][j]
                strong_edge[i][j] = sobel[i][j]
                weak_edge[i][j] = 0
            elif sobel[i][j] >= minThresh:
                canny[i][j] = sobel[i][j]
                weak_edge[i][j] = sobel[i][j]
                strong_edge[i][j] = 0
            else:
                canny[i][j] = 0
                strong_edge[i][j] = 0
                weak_edge[i][j] = 0
    
    # Connected Component Analysis
    neighbor_thresh = 2
    for i in range(weak_edge.shape[0]):
        for j in range(weak_edge.shape[1]):
            neighbors = 0
            if weak_edge[i][j] == 0:
                continue
            # check for corner
            if i == 0 and j == 0:
                if strong_edge[1][0] != 0:       neighbors += 1
                if strong_edge[1][1] != 0:       neighbors += 1
                if strong_edge[0][1] != 0:       neighbors += 1
            elif i == weak_edge.shape[0] - 1 and j == 0:
                if strong_edge[i-1][0] != 0:     neighbors += 1
                if strong_edge[i-1][1] != 0:     neighbors += 1
                if strong_edge[i][1] != 0:       neighbors += 1
            elif i == 0 and j == weak_edge.shape[1] - 1:
                if strong_edge[i][j-1] != 0:     neighbors += 1
                if strong_edge[i+1][j-1] != 0:   neighbors += 1
                if strong_edge[i+1][j] != 0:     neighbors += 1
            elif i == weak_edge.shape[0] - 1 and j == weak_edge.shape[1] - 1:
                if strong_edge[i-1][j] != 0:     neighbors += 1
                if strong_edge[i-1][j-1] != 0:   neighbors += 1
                if strong_edge[i][j-1] != 0:     strong_edge += 1
            # check for edge
            elif i == 0:
                if strong_edge[i][j-1] != 0:     neighbors += 1
                if strong_edge[i+1][j-1] != 0:   neighbors += 1
                if strong_edge[i+1][j] != 0:     neighbors += 1
                if strong_edge[i+1][j+1] != 0:   neighbors += 1
                if strong_edge[i][j+1] != 0:     neighbors += 1
            elif i == weak_edge.shape[0] - 1:
                if strong_edge[i][j-1] != 0:     neighbors += 1
                if strong_edge[i-1][j-1] != 0:   neighbors += 1
                if strong_edge[i-1][j] != 0:     neighbors += 1
                if strong_edge[i-1][j+1] != 0:   neighbors += 1
                if strong_edge[i][j+1] != 0:     neighbors += 1
            elif j == 0:
                if strong_edge[i-1][j] != 0:     neighbors += 1
                if strong_edge[i-1][j+1] != 0:   neighbors += 1
                if strong_edge[i][j+1] != 0:     neighbors += 1
                if strong_edge[i+1][j+1] != 0:   neighbors += 1
                if strong_edge[i+1][j] != 0:     neighbors += 1
            elif j == weak_edge.shape[1] - 1:
                if strong_edge[i-1][j] != 0:     neighbors += 1
                if strong_edge[i-1][j-1] != 0:   neighbors += 1
                if strong_edge[i][j-1] != 0:     neighbors += 1
                if strong_edge[i+1][j-1] != 0:   neighbors += 1
                if strong_edge[i+1][j] != 0:     neighbors += 1
            # check for the 8 neighboring strong edge pixels
            else:
                if strong_edge[i-1][j-1] != 0:   neighbors += 1
                if strong_edge[i-1][j] != 0:     neighbors += 1
                if strong_edge[i-1][j+1] != 0:   neighbors += 1
                if strong_edge[i][j+1] != 0:     neighbors += 1
                if strong_edge[i+1][j+1] != 0:   neighbors += 1
                if strong_edge[i+1][j] != 0:     neighbors += 1
                if strong_edge[i+1][j-1] != 0:   neighbors += 1
                if strong_edge[i][j-1] != 0:     neighbors += 1
            # supress if no strong edges in neihborhood
            if neighbors < neighbor_thresh: canny[i][j] = 0
        
    canny = cv2.erode(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)), iterations=1)
    
    return canny

canny_implemented = canny_experimental(gray)
fig.add_subplot(1, 2, 2)
plt.imshow(canny_implemented, cmap='gray')

plt.show()

