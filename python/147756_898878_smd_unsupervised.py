get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")

get_ipython().run_line_magic('matplotlib', 'inline')
from ml import plots
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
pd.options.display.max_rows = 10

discrete_cmap = LinearSegmentedColormap.from_list('discrete', colors = [(0.8, 0.2, 0.3), (0.98, 0.6, 0.02), (0.1, 0.8, 0.3), (0, 0.4, 0.8), ], N=4)

plots.set_plot_style()

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data  # we only take the first two features.
y = iris.target

f, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(3, 2)
ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=discrete_cmap, label='species')
ax2.scatter(X[:, 0], X[:, 2], c=y, cmap=discrete_cmap)
ax3.scatter(X[:, 0], X[:, 3], c=y, cmap=discrete_cmap)
ax4.scatter(X[:, 1], X[:, 2], c=y, cmap=discrete_cmap)
ax5.scatter(X[:, 1], X[:, 3], c=y, cmap=discrete_cmap)
ax6.scatter(X[:, 2], X[:, 3], c=y, cmap=discrete_cmap)

from sklearn.datasets import load_boston
houses = load_boston()
names = list(houses.feature_names) +  ['price']
data = pd.DataFrame(data=np.c_[houses['data'], houses['target']], columns=names)

sns.pairplot(data[['RM', 'NOX', 'B', 'price']])

sns.jointplot(data.RM, data.price, kind='scatter')

np.random.seed(1234)
from sklearn.datasets import make_blobs

# choose a random number between 1 and 3
k = np.random.randint(1, 4)

# create k blobs 
X, y = make_blobs(n_samples=300, centers=k, center_box=(-2, 2), cluster_std=0.5)

# plot points without any color coding.
plt.scatter(X[:, 0], X[:, 1])
plt.axis('off')
None

#plot the same points this time with color coding 
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=discrete_cmap)
plt.axis('off')
None

np.random.seed(1234)
from sklearn.cluster import KMeans

# create three random blobs
X, y = make_blobs(n_samples=300, centers=3, center_box=(-2, 2), cluster_std=0.5)

# use KMeans to predict clusters for each sample in X
prediction = KMeans(n_clusters=3).fit_predict(X)

# shift labels to get the right colors
prediction = np.choose(prediction, [3, 0, 2])

# plot rings with predicted clusters
plt.scatter(X[:, 0], X[:, 1], facecolor='', edgecolors=discrete_cmap(prediction), lw=2,  s=380, label='prediction')

# plot points with true cluster associations
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=discrete_cmap, label='truth')
plt.legend(loc='upper right')
plt.axis('off')
None

from IPython import display

# create 4 clsuters
k = 4

# choose inital cluster centers
X, y = make_blobs(n_samples=500, centers=k, center_box=(-2, 2), cluster_std=.5, random_state=1234)

fig = plt.figure()
ax = fig.add_subplot(111)

init_centers = np.array([[0, 0], [1, 1], [1, 2], [1, 3]])

# loop over each iteration. do it 5 times
for i in np.tile(range(1, 15), 5):
    kmeans = KMeans(n_clusters=4, init=init_centers, max_iter=i).fit(X)
    
    ax.cla()
    ax.set_title('Iteration {}'.format(i))
    
    ax.scatter(X[:,0], X[:,1],c=kmeans.labels_, cmap=discrete_cmap)
    ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black', marker='h', s=400, alpha=0.3,  label='cluster center')
    
    ax.legend(loc=3)
    ax.axis('off')
    
    display.clear_output(wait=True)
    display.display(plt.gcf())

plt.close()
    

u1 = np.random.uniform(-1, 0, size=(500, 2))
u2 = np.random.uniform(0, 1, size=(500, 2))
X = np.append(u1, u2, axis=0)
plt.scatter(X[:, 0], X[:, 1], color='gray')
plt.axis('off')
None

# use KMeans to predict clusters for each sample in X using the deliberatley wrong value of k=4
prediction = KMeans(n_clusters=4).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=prediction, cmap=discrete_cmap)
plt.axis('off')
None

# use KMeans to predict clusters for each sample in X this time using the correct value for k=2
prediction = KMeans(n_clusters=2).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=prediction, cmap=discrete_cmap)
plt.axis('off')
None

X, y = make_blobs(n_samples=1300, centers=2, random_state=2)
transformation = [[20, 0], [0, 1]]
X = np.dot(X, transformation)
prediction = KMeans(n_clusters=2,).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=prediction, cmap=discrete_cmap)
plt.axes().set_aspect('equal', 'datalim')
None

from sklearn import preprocessing

f, axs = plt.subplots(4, 1, figsize=(10, 20), sharex=True)

scalers=[
        preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(feature_range=(-1, 1)),
        preprocessing.MaxAbsScaler(),
        preprocessing.QuantileTransformer()
]

for scaler, ax in zip(scalers, axs):
    X_prime = scaler.fit_transform(X)
    prediction = KMeans(n_clusters=2, random_state=0).fit_predict(X_prime)
    ax.set_title(scaler.__class__.__name__)
    ax.scatter(X_prime[:, 0], X_prime[:, 1], c=prediction, cmap=discrete_cmap)
    
None

from sklearn.datasets import load_sample_image
from skimage import color

#load an example image as a 3D array
image = load_sample_image("flower.jpg")
image = np.array(image, dtype=np.float64) / 255

#store width length and number of colors
width, length, d = image.shape  

# show image
plt.imshow(image)

#convert image to hsv for nicer plots
image_hsv = color.rgb2hsv(image)

from mpl_toolkits import mplot3d

# plot H, V and S values of each pixel into a 3D coordinate system.

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter3D(image_hsv[:, :, 0], image_hsv[:, :, 1], image_hsv[:, :, 2], c=image_hsv.reshape(-1, 3), alpha=0.5)
None

from sklearn.utils import resample

# perform k-means on a small random sample of pixels
# we could use all pixels here but it would take much too long.
# we reshape the image into a 2D array for resampling and k-means fitting
flattened_image = image_hsv.reshape(-1, 3)
sample = resample(flattened_image, n_samples=1000, replace=False)

# get the desired number of clsuter centers
kmeans = KMeans(n_clusters = 50).fit(sample)
centroids = kmeans.cluster_centers_

# sort centroids by hue to visualize the new colors.
idx = np.argsort(centroids[:, 0])

plt.figure(figsize=(12, 2))
plt.scatter(np.linspace(0, 1, len(idx)), np.ones_like(idx),  c=centroids[idx], s=10000)
plt.axis('off')
None

# associate each pixel with the number of a cluster (i.e. the nearest centroid/color)
labels = kmeans.predict(flattened_image)

# get the actual value for each cluster centroid and reshape into a 3D image
reconstructed  = centroids[labels].reshape(width, length, d)

# convert to RGB and plot again.
plt.imshow(color.hsv2rgb(reconstructed))

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=400, noise=0.1)
prediction = KMeans(n_clusters=2, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=prediction, cmap=discrete_cmap)
plt.axis('off')
None

from sklearn.datasets import make_moons, make_checkerboard, make_circles
from sklearn.cluster import DBSCAN

f, [top, center, bottom] = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(20, 20))

X_moon, y_moon = make_moons(noise=0.05, random_state=0)
X_circle, y_circle = make_circles(noise=0.05, factor=0.4, random_state=0, n_samples=200)
X_blobs, y_blobs = make_blobs(centers=2, center_box=(-0.5, 0.5), cluster_std=0.4, random_state=0)
X_long, y_long = make_blobs(centers=2, center_box=(-2.1, 2.1), cluster_std=0.1, random_state=0)

data = [(X_moon, y_moon), (X_long, y_long), (X_circle, y_circle), (X_blobs, y_blobs)]

for ax, (X, y) in zip(top, data):
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=discrete_cmap)
    ax.axis('off')
    
for ax, (X, y) in zip(center, data):
    prediction = KMeans(n_clusters=2, random_state=0).fit_predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=prediction, cmap=discrete_cmap)
    ax.axis('off')
    
for ax, (X, y) in zip(bottom, data):
    prediction = DBSCAN(eps=0.339).fit_predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=prediction, cmap=discrete_cmap)
    ax.axis('off')
    
top[1].set_title('True Clusters')
center[1].set_title('K - Mean Clusters')
bottom[1].set_title('DBSCAN Clusters')
None

from sklearn.metrics import silhouette_score

X, y = make_moons(n_samples=360, noise=0.09, random_state=172)

km = KMeans(n_clusters=2)
prediction_kmeans = km.fit_predict(X)
score_kmeans = silhouette_score(X, km.labels_ ) 

db = DBSCAN(eps=0.18)
prediction_db = db.fit_predict(X)
score_db = silhouette_score(X, db.labels_ ) 

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title('k-Means clustering score: {:0.3f}'.format(score_kmeans))
ax1.scatter(X[:, 0], X[:, 1], c=prediction_kmeans, cmap=discrete_cmap)
ax1.axis('off')

ax2.set_title('DBSCAN clustering: {:0.3f}'.format(score_db))
ax2.scatter(X[:, 0], X[:, 1], c=prediction_db, cmap=discrete_cmap)
ax2.axis('off')
None



