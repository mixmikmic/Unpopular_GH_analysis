import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

import matplotlib.image as mpimg
img = mpimg.imread('apple_tree.png')
nrows, ncols, depth = img.shape

top_row = img[0]
top_left_pixel = top_row[0]
red, green, blue = top_left_pixel
print red, green, blue

plt.imshow(img)

from sklearn.cluster import KMeans

X = img.reshape((nrows * ncols, depth))
km = KMeans(n_clusters=3, init='k-means++', n_init=25)
km = km.fit(X)

rgb = km.cluster_centers_
print rgb

y = km.predict(X).reshape((nrows, ncols))

for i in xrange(nrows):
    for j in xrange(ncols):
        img[i, j] = rgb[y[i, j]]

plt.imshow(img)

