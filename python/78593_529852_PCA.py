# install relevant modules
import numpy as np
import pandas as pd

# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 20})

import seaborn as sns
sns.set(style="white", color_codes=True)

# Read file
df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.head(2)

# split data table into data X and class labels y

X = df.ix[:,0:4].values
y = df.ix[:,4].values

print(X[:1, :])
print(y[:1])

import seaborn as sns

#g = sns.FacetGrid(df, hue="class", col=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid'], col_wrap=4, sharex=False)

g = sns.FacetGrid(df, col="class", size=5, aspect = 1.2)
g.map(sns.kdeplot, "sepal_len", shade=True).fig.subplots_adjust(wspace=.3)
g.map(sns.kdeplot, "petal_len", shade=True, color = "r").fig.subplots_adjust(wspace=.3)
g.map(sns.kdeplot, "petal_wid", shade=True, color = "g").fig.subplots_adjust(wspace=.3)
g.map(sns.kdeplot, "sepal_wid", shade=True, color = "m").fig.subplots_adjust(wspace=.3)
sns.set(font_scale=2)

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

X_std[:5]

# Calculate covariance matrix
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

# Could alternatively use:
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

# Eigendecomposition on the covariance matrix:

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# Eigendecomposition on the Correlation matrix:

cor_mat2 = np.corrcoef(X.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

u,s,v = np.linalg.svd(X_std.T)
u

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

# Explained variance
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
#pc_col = ['PC1', 'PC2', 'PC3', 'PC4']
index = [1, 2, 3, 4]
plt.bar(pc_col, var_exp)
plt.xticks(index , ('PC1', 'PC2', 'PC3', 'PC4'))
plt.plot(index , cum_var_exp, c = 'r')
plt.ylabel('Explained variance')
plt.show()

matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)

Y = X_std.dot(matrix_w)
Y[:3]

y[:5]

from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(X_std)

Y_sklearn[:3]
new_list = []
for row in range(len(y)):
    if y[row]=='Iris-setosa':
        new_list.append([Y_sklearn[:, 0], Y_sklearn[:, 1], 0, 'Iris-setosa'])
    elif y[row]=='Iris-versicolor':
        new_list.append([Y_sklearn[:, 0], Y_sklearn[:, 1], 0, 'Iris-versicolor'])
    else:
        new_list.append([Y_sklearn[:, 0], Y_sklearn[:, 1], 0, 'Iris-verginica'])
plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1])
plt.show()



