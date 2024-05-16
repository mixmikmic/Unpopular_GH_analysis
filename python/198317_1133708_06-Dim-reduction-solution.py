from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

print(X[:5])

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

pca = PCA(n_components=2)
tsne = TSNE(n_components=2)

X_pca = pca.fit(X).transform(X)
X_tsne = tsne.fit_transform(X)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

#plot the PCA dimensions
plt.figure(figsize=(10,6))

for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], alpha=.8, label=target_name)
plt.legend(loc='lower right')
plt.title('PCA of the Iris dataset.')
plt.show()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

#plot the t-SNE dimensions
plt.figure(figsize=(10,6))
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], alpha=0.8,label=target_name)
plt.legend(loc='best')
plt.title('t-SNE of the Iris dataset.')
plt.show()

