import csv
import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from IPython.display import Image

print('csv: {}'.format(csv.__version__))
print('numpy: {}'.format(np.__version__))
print('scipy: {}'.format(sp.__version__))
print('pandas: {}'.format(pd.__version__))
print('sklearn: {}'.format(sk.__version__))

Image(url='http://www.radford.edu/~rsheehy/Gen_flash/Tutorials/Linear_Regression/reg-tut_files/linreg3.gif')

filename = '/Users/jessicagronski/Downloads/bldgstories1.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = np.array(x).astype('float')

# Load CSV with numpy
import numpy
raw_data = open(filename, 'rb')
data = numpy.loadtxt(raw_data, delimiter=",")

# Load CSV using Pandas
import pandas
colnames = ['year', 'height', 'stories']
data = pandas.read_csv(filename, names=colnames)
data = pandas.DataFrame(data, columns=colnames)

print('Dimensions:')
print(data.shape)
print('Ten observations:')
print(data.head(6))
print('Correlation matrix:')
correlations = data.corr(method='pearson')
print(correlations)

pandas.set_option('precision', 3)
description = data.describe()
print(description)

from sklearn import linear_model
obj = linear_model.LinearRegression()
obj.fit(np.array(data.height.values.reshape(-1,1)), data.stories )#need this values.reshape(-1,1) to avoid deprecation warnings
print( obj.coef_, obj.intercept_ )

x_min, x_max = data.height.values.min() - .5, data.height.values.max() + .5 # for plotting
x_rng = np.linspace(x_min,x_max,200)

plt.plot(x_rng, x_rng * obj.coef_ + obj.intercept_, 'k')
plt.plot(data.height.values, data.stories.values,'ro', alpha = 0.5)
plt.show()

obj2 = linear_model.LinearRegression()
X = np.array( (data.height.values, data.year.values))
obj2.fit(X.transpose() , data.stories)
print(obj2.coef_, obj2.intercept_)

from mpl_toolkits.mplot3d import Axes3D

ax = plt.axes(projection = '3d')
#ax.plot(data.height.values, data.year.values , data.stories.values, 'bo')

ax.plot_surface(data.height.values, data.year.values, (np.dot(X.transpose(),obj2.coef_)                 + obj2.intercept_), color='b')

ax.show()
#plt.close()

##### doesn't work - have the students try to solve it.

print(np.dot(X.transpose(),obj2.coef_).shape)

data.height.values.shape

