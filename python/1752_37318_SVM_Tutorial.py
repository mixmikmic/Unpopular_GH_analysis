# Print figures in the notebook
get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets # Import the dataset from scikit-learn
from sklearn.svm import SVC

# Import patch for drawing rectangles in the legend
from matplotlib.patches import Rectangle

# Create color maps
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

# Create a legend for the colors, using rectangles for the corresponding colormap colors
labelList = []
for color in cmap_bold.colors:
    labelList.append(Rectangle((0, 0), 1, 1, fc=color))

# Import some data to play with
iris = datasets.load_iris()

# Store the labels (y), label names, features (X), and feature names
y = iris.target       # Labels are stored in y as numbers
labelNames = iris.target_names # Species names corresponding to labels 0, 1, and 2
X = iris.data
featureNames = iris.feature_names

# Plot the data

# Sepal length and width
X_small = X[:,:2]
# Get the minimum and maximum values with an additional 0.5 border
x_min, x_max = X_small[:, 0].min() - .5, X_small[:, 0].max() + .5
y_min, y_max = X_small[:, 1].min() - .5, X_small[:, 1].max() + .5

plt.figure(figsize=(8, 6))

# Plot the training points
plt.scatter(X_small[:, 0], X_small[:, 1], c=y, cmap=cmap_bold)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Sepal width vs length')

# Set the plot limits
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Plot the legend
plt.legend(labelList, labelNames)

plt.show()

# Create an instance of SVM and fit the data.
clf = SVC(kernel='linear', decision_function_shape='ovo')
clf.fit(X_small, y)

h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X_small[:, 0].min() - 1, X_small[:, 0].max() + 1
y_min, y_max = X_small[:, 1].min() - 1, X_small[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # Make a prediction oat every point 
                                               # in the mesh in order to find the 
                                               # classification areas for each label

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X_small[:, 0], X_small[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (SVM)")
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')

# Plot the legend
plt.legend(labelList, labelNames)

plt.show()

# Add our new data examples
examples = [[4.3, 2.5], # Plant A
            [6.3, 2.1]] # Plant B


# Create an instance of SVM and fit the data
clf = SVC(kernel='linear', decision_function_shape='ovo')
clf.fit(X_small, y)

# Predict the labels for our new examples
labels = clf.predict(examples)

# Print the predicted species names
print('A: ' + labelNames[labels[0]])
print('B: ' + labelNames[labels[1]])

# Now plot the results
h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X_small[:, 0].min() - 1, X_small[:, 0].max() + 1
y_min, y_max = X_small[:, 1].min() - 1, X_small[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # Make a prediction oat every point 
                                               # in the mesh in order to find the 
                                               # classification areas for each label

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X_small[:, 0], X_small[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (SVM)")
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')

# Display the new examples as labeled text on the graph
plt.text(examples[0][0], examples[0][1],'A', fontsize=14)
plt.text(examples[1][0], examples[1][1],'B', fontsize=14)

# Plot the legend
plt.legend(labelList, labelNames)

plt.show()

def plot_svc_decision_function(clf):
    """Plot the decision function for a 2D SVC"""
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
    Y, X = np.meshgrid(y, x)
    P = np.zeros((3,X.shape[0],X.shape[1]))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            P[:, i,j] = clf.decision_function([[xi, yj]])[0]
    for ind in range(3):
        plt.contour(X, Y, P[ind,:,:], colors='k',
                levels=[-1, 0, 1],
                linestyles=['--', '-', '--'])

# Now plot the results
h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X_small[:, 0].min() - 1, X_small[:, 0].max() + 1
y_min, y_max = X_small[:, 1].min() - 1, X_small[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # Make a prediction at every point 
                                               # in the mesh in order to find the 
                                               # classification areas for each label

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X_small[:, 0], X_small[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (SVM)")
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')

# Display the new examples as labeled text on the graph
plt.text(examples[0][0], examples[0][1],'A', fontsize=14)
plt.text(examples[1][0], examples[1][1],'B', fontsize=14)

# Plot the legend
plt.legend(labelList, labelNames)

plot_svc_decision_function(clf) # Plot the decison function

plt.show()

# Create an instance of SVM and fit the data.
clf = SVC(kernel='rbf', decision_function_shape='ovo') # Use the RBF kernel this time
clf.fit(X_small, y)

# Now plot the results
h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X_small[:, 0].min() - 1, X_small[:, 0].max() + 1
y_min, y_max = X_small[:, 1].min() - 1, X_small[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # Make a prediction oat every point 
                                               # in the mesh in order to find the 
                                               # classification areas for each label

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X_small[:, 0], X_small[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (SVM)")
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')

# Display the new examples as labeled text on the graph
plt.text(examples[0][0], examples[0][1],'A', fontsize=14)
plt.text(examples[1][0], examples[1][1],'B', fontsize=14)

# Plot the legend
plt.legend(labelList, labelNames)

plt.show()


# Create an instance of SVM and fit the data.
clf = SVC(kernel='rbf', decision_function_shape='ovo') # Use the RBF kernel this time
clf.fit(X_small, y)

# Now plot the results
h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X_small[:, 0].min() - 1, X_small[:, 0].max() + 1
y_min, y_max = X_small[:, 1].min() - 1, X_small[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # Make a prediction oat every point 
                                               # in the mesh in order to find the 
                                               # classification areas for each label

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X_small[:, 0], X_small[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (SVM)")
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')

# Display the new examples as labeled text on the graph
plt.text(examples[0][0], examples[0][1],'A', fontsize=14)
plt.text(examples[1][0], examples[1][1],'B', fontsize=14)

# Plot the legend
plt.legend(labelList, labelNames)

plot_svc_decision_function(clf) # Plot the decison function

plt.show()


# Your code here!

# Your code here!



