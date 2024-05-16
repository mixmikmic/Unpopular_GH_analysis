import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

for clr, cls in zip(['white', 'red', 'green'], np.unique(y)):
    plt.plot(X[y == cls, 0], X[y == cls, 1], 'o', color=clr)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.xlim(0, 8)
plt.ylim(0, 3)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
X_std = stdsc.transform(X)

from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

param_grid = dict(C=np.logspace(-3, 2, base=10))
grid = GridSearchCV(estimator=SVC(kernel='linear'), param_grid=param_grid, cv=10, scoring='accuracy')
grid.fit(X_train_std, y_train)
print grid.best_score_
print grid.best_params_

svm = SVC(kernel='linear', C=grid.best_params_['C'])
svm.fit(X_train_std, y_train)

svm.score(X_test_std, y_test)

fig, ax = plt.subplots(nrows=1, ncols=1)

# decision boundary plot
x_min, x_max = -2.5, 2.5
y_min, y_max = -2.5, 2.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=100), np.linspace(y_min, y_max, num=100))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]) 
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
ax.set_xlabel("Pedal length (standardized)")
ax.set_ylabel("Pedal width (standardized)")
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)

# original data
for clr, cls in zip(['white', 'red', 'green'], np.unique(y)):
    ax.scatter(x=X_std[y == cls, 0], y=X_std[y == cls, 1], marker='o', c=clr, s=75)

X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:,0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.plot([-4, 4], [0, 0], 'k:')
plt.plot([0, 0], [-4, 4], 'k:')
plt.plot(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], 'wo')
plt.plot(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], 'ro')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

X_xor_std = stdsc.fit_transform(X_xor)
param_grid = dict(C=np.logspace(-3, 2, base=10), gamma=np.logspace(-3, 2, base=10))
grid = GridSearchCV(estimator=SVC(kernel='rbf'), param_grid=param_grid, cv=10, scoring='accuracy')
grid.fit(X_xor_std, y_xor)
print grid.best_score_
print grid.best_params_

svm = SVC(kernel='rbf', C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
svm.fit(X_xor_std, y_xor)

fig, ax = plt.subplots(nrows=1, ncols=1)

# decision boundary plot
x_min, x_max = -4, 4
y_min, y_max = -4, 4
xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=100), np.linspace(y_min, y_max, num=100))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]) 
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
ax.set_xlabel("Feature 1 (standardized)")
ax.set_ylabel("Feature 2 (standardized)")
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

# original data
colors = ['white', 'red']
for idx, cls in enumerate(np.unique(y_xor)):
    ax.scatter(x=X_xor_std[y_xor == cls, 0], y=X_xor_std[y_xor == cls, 1], marker='o', c=colors[idx], s=75)

from sklearn.linear_model import SGDClassifier
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')

