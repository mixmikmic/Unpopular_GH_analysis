from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

print(X[:5])

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

k = 3
kmeans = KMeans(n_clusters=k).fit(X)
agglomerative = AgglomerativeClustering(n_clusters=k).fit(X)
y_kmeans = kmeans.labels_
y_agglomerative = agglomerative.labels_

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

#plot the k-means clusters
plt.figure(figsize=(10,6))
for i in range(k):
    plt.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1], alpha=.8)
plt.title('k-means clustering of the Iris dataset.')
plt.legend(loc='lower right')
plt.show()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

#plot the k-means clusters
plt.figure(figsize=(10,6))
for i in range(k):
    plt.scatter(X_pca[y_agglomerative == i, 0], X_pca[y_agglomerative == i, 1], alpha=.8)
plt.title('Agglomerative clustering of the Iris dataset.')
plt.legend(loc='lower right')
plt.show()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

#plot the PCA dimensions
plt.figure(figsize=(10,6))
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], alpha=.8, label=target_name)
plt.legend(loc='best')
plt.title('True target values of the Iris dataset.')
plt.legend(loc='lower right')
plt.show()

import itertools
from sklearn.metrics import accuracy_score

kmeans_accuracies = []
agglomerative_accuracies = []
for permutation in itertools.permutations([0, 1, 2]):    
    kmeans_accuracies.append(accuracy_score(y, list(map(lambda x: permutation[x], y_kmeans))))
    agglomerative_accuracies.append(accuracy_score(y, list(map(lambda x: permutation[x], y_agglomerative))))
    
print ('Accuracy of k-means clustering: {}'.format(max(kmeans_accuracies)))
print ('Accuracy of agglomerative clustering: {}'.format(max(agglomerative_accuracies)))

