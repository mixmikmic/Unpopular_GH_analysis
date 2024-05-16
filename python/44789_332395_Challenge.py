#Import all the libraries
import pandas as pd
import numpy as np
from sklearn import linear_model as model
import matplotlib.pyplot as plt 

#read data from the challenge_dataset
dataframe = pd.read_csv('input/challenge_dataset.txt')
x_values = dataframe[[0]]
y_values = dataframe[[1]]

#train model on data
regr = model.LinearRegression()
regr.fit(x_values, y_values)

# The coefficients
print('Coefficients: ', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f ' % np.mean((regr.predict(x_values) - y_values) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_values, y_values))

#Visualize Results
plt.scatter(x_values, y_values)
plt.plot(x_values, regr.predict(x_values))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Challenge Dataset')
plt.show()

get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
from sklearn import linear_model as model
from sklearn import datasets
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
# Imports
import matplotlib as mpl

# Loading data
#read data from the challenge_dataset
iris = datasets.load_iris()
#print(iris.data.shape)
# for the above bonus, consider only 3 different vaiables 
# i.e. two are input variable and one is output variable
x_values = iris.data[:,1:3]
print(x_values.shape)
y_values = iris.target

#train model on data
linearmodel = model.LinearRegression()
linearmodel.fit(x_values, y_values)


# The coefficients
print('Coefficients: ', linearmodel.coef_)
# The mean squared error
print('Mean squared error: %.2f ' % np.mean((linearmodel.predict(x_values) - y_values) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % linearmodel.score(x_values, y_values))


#Visualize Results
fig = plt.figure()
fig.set_size_inches(12.5,7.5)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_values[:,0],x_values[:,1], y_values, c='g', marker= 'o')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Species')
ax.set_title('Orignal Dataset')
ax.view_init(10, -45)

fig1 = plt.figure()
fig1.set_size_inches(12.5,7.5)
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(x_values[:,0],x_values[:,1], linearmodel.predict(x_values), c='r', marker= 'o')
#ax.plot_surface(x_values[:,0],x_values[:,1], linearmodel.predict(x_values), cmap=cm.hot, color='b', alpha=0.2); 
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Species')
ax.set_title('Predicted Dataset')
ax.view_init(10, -45)

