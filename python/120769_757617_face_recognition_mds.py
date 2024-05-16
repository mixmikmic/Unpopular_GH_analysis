get_ipython().magic('matplotlib inline')
from time import time
import logging
import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import manifold

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

accuracies = []
components = []
for nn in xrange(2,30,1):
    n_components = nn
    mds = manifold.MDS(n_components, max_iter=100, n_init=1)

    X_train_changed = mds.fit_transform(X_train)
    X_test_changed = mds.fit_transform(X_test)
    param_grid = {'C': [1,1e1,1e2,5e2,1e3, 5e3, 1e4, 5e4, 1e5],
                          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_changed, y_train)
    y_pred = clf.predict(X_test_changed)
    accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
    components.append(n_components)
    print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    print(classification_report(y_test, y_pred, target_names=target_names))

plt.plot(components,accuracies)
plt.title('Number of Components vs Accuracy')
plt.xlabel('Components')
plt.ylabel('Accuracy')
plt.show()

accuracies = []
components = []
for nn in xrange(2,200,10):
    n_components = nn
    mds = manifold.MDS(n_components, max_iter=100, n_init=1)

    X_train_changed = mds.fit_transform(X_train+1)
    X_test_changed = mds.fit_transform(X_test+1)
    param_grid = {'C': [1,1e1,1e2,5e2,1e3, 5e3, 1e4, 5e4, 1e5],
                          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_changed, y_train)
    y_pred = clf.predict(X_test_changed)
    accuracies.append(float(np.sum(y_test==y_pred))/len(y_pred))
    components.append(n_components)
    print('For '+str(n_components)+' components, accuracy is '+str(float(np.sum(y_test==y_pred))/len(y_pred))+' confusion matrix is: ')
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    print(classification_report(y_test, y_pred, target_names=target_names))

plt.plot(components,accuracies)
plt.title('Number of Components vs Accuracy')
plt.xlabel('Components')
plt.ylabel('Accuracy')
plt.show()

