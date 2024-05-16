import cv2
import numpy as np
from matplotlib import pyplot as plt

# Matplotlib uses RGB whereas CV uses BGR
def inverted(image):
    inv_image = np.zeros_like(image)
    inv_image[:,:,0] = image[:,:,2]
    inv_image[:,:,1] = image[:,:,1]
    inv_image[:,:,2] = image[:,:,0]
    return inv_image


image = cv2.imread("../resources/messi.jpg")

# Draw a line
cv2.line(image, (40, 40), (100, 200), (255, 0, 0), 2)

# Draw a rectangle
cv2.rectangle(image, (300, 300), (200, 250), (0, 0, 255), 3)

# Draw a circle
cv2.circle(image, (100, 200), 50, (0, 255, 0), 4)

# Draw an ellipse
cv2.ellipse(image, (300, 100), (100, 80), 45, 0, 360, (255, 255, 255), 5)

plt.imshow(inverted(image))
plt.show()

# Drawing a polygon

image = cv2.imread("../resources/messi.jpg")

points = np.array([
    [10,10],
    [100, 200],
    [200, 300],
    [300, 200],
    [300, 10]
])

cv2.polylines(image, [points], True, (0, 255, 0), 4)

plt.imshow(inverted(image))
plt.show()

# Filling the polygon
cv2.fillConvexPoly(image, points, (100, 100, 100))

plt.imshow(inverted(image))
plt.show()

# Filling multiple polygons
image = cv2.imread("../resources/messi.jpg")

triangle = np.array([
    [10, 10],
    [10, 50],
    [60, 10]
])

square = np.array([
    [100, 100],
    [100, 200],
    [200, 200],
    [200, 100]
])

cv2.fillPoly(image, [triangle, square], (200, 200, 0))

# Writing text
cv2.putText(image, "MESSI", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

plt.imshow(inverted(image))
plt.show()

image = cv2.imread("../resources/messi.jpg")

# flags and params will not be used, they are passed by opencv itself, if any
def mouse_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 10, (200, 200, 0), 3)


cv2.namedWindow('testing')
cv2.setMouseCallback('testing', mouse_click)
        
while True:
    cv2.imshow('testing', image)
    k = cv2.waitKey(10) & 255
    if k == 27:
        cv2.destroyWindow('testing')
        break

plt.imshow(inverted(image))
plt.show()

