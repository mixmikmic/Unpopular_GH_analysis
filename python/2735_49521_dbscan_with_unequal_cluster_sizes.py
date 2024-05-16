import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Generate equipartitioned sample data
centers = [[0, 0], [20, 18], [18, 20]]

X, labels_true = make_blobs(n_samples=600, 
                            centers=centers, 
                            cluster_std=0.4, 
                            random_state=0)

# Generate 10x more data for one cluster
extra_X0, labels_true_extra = make_blobs(n_samples=6000, 
                                    centers=centers[0], 
                                    cluster_std=0.4, 
                                    random_state=0)

# Combine the datasets
X = np.vstack([X, extra_X0])
labels_true = np.concatenate([labels_true, labels_true_extra])


# Scale all the data
X = StandardScaler().fit_transform(X) 

# Plot the data
xx, yy = zip(*X)
plt.scatter(xx, yy)
plt.show()

db = DBSCAN(eps=0.15, min_samples=50).fit(X)
core_samples = db.core_sample_indices_
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print n_clusters_

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

get_ipython().magic('timeit db = DBSCAN(eps=0.15, min_samples=50).fit(X)')



