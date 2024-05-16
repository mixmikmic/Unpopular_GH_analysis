# Print figures in the notebook
get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets # Import datasets from scikit-learn

# Import patch for drawing rectangles in the legend
from matplotlib.patches import Rectangle

# Create color maps
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Create a legend for the colors, using rectangles for the corresponding colormap colors
labelList = []
for color in cmap_bold.colors:
    labelList.append(Rectangle((0, 0), 1, 1, fc=color))

# Import some data to play with
iris = datasets.load_iris()

# List the data keys
print('Keys: ' + str(iris.keys()))
print('Label names: ' + str(iris.target_names))
print('Feature names: ' + str(iris.feature_names))
print('')

# Store the labels (y), label names, features (X), and feature names
y = iris.target       # Labels are stored in y as numbers
labelNames = iris.target_names # Species names corresponding to labels 0, 1, and 2
X = iris.data
featureNames = iris.feature_names

# Show the first five examples
print(iris.data[1:5,:])

# Plot the data

# Sepal length and width
X_sepal = X[:,:2]
# Get the minimum and maximum values with an additional 0.5 border
x_min, x_max = X_sepal[:, 0].min() - .5, X_sepal[:, 0].max() + .5
y_min, y_max = X_sepal[:, 1].min() - .5, X_sepal[:, 1].max() + .5

plt.figure(figsize=(8, 6))

# Plot the training points
plt.scatter(X_sepal[:, 0], X_sepal[:, 1], c=y, cmap=cmap_bold)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Sepal width vs length')

# Set the plot limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.legend(labelList, labelNames)

plt.show()

# Put your code here!

from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

print('Original dataset size: ' + str(X.shape))
print('Training dataset size: ' + str(X_train.shape))
print('Test dataset size: ' + str(X_test.shape))



