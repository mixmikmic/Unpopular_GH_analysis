import math
import warnings

from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import statsmodels.formula.api as smf

# Display preferences.
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.float_format = '{:.3f}'.format

# Suppress annoying harmless error.
warnings.filterwarnings(
    action="ignore",
    module="scipy",
    message="^internal gelsd"
)

# Acquire, load, and preview the data.
data = pd.read_csv('Advertising.csv')
display(data.head())

# Instantiate and fit our model.
regr = linear_model.LinearRegression()
Y = data['Sales'].values.reshape(-1, 1)
X = data[['TV','Radio','Newspaper']]
# Square TV radio, and newspaper in order to achieve homoscedasticity.
data['TV Sqrd'] = data['TV'] * data['TV']
data['Radio Sqrd'] = data['Radio'] * data['Radio']
data['Newspaper Sqrd'] = data['Newspaper'] * data['Newspaper']
X2 = data[['TV Sqrd', 'Radio Sqrd', 'Newspaper Sqrd']]
regr.fit(X2, Y)
data['Sales Sqrd'] = data['Sales'] * data['Sales']
Y2 = data['Sales Sqrd'].values.reshape(-1, 1) 

# Inspect the results.
print('\nCoefficients: \n', regr.coef_)
print('\nIntercept: \n', regr.intercept_)
print('\nR-squared:')
print(regr.score(X2, Y))

# I will look for outliers in my data and then remove them. 
print(data.describe())
# It seems like the biggest problem is somewhere in tv since it has the highest sd. 
#data[data['TV'] > 290]
# 30, 35, 42, 101

data = data.drop([30, 35, 42, 101])

Y3 = data['Sales'].values.reshape(-1, 1)
X3 = data[['TV','Radio','Newspaper']]

# Extract predicted values.
predicted = regr.predict(X3).ravel()
actual = data['Sales']

# Calculate the error, aka residual.
residual = actual - predicted


plt.hist(residual)
plt.title('Residual counts')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.show()

plt.scatter(predicted, residual)
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.axhline(y=0)
plt.title('Residual vs. Predicted')
plt.show()

# Hm... looks a bit concerning.

# Your code here.
corrmat = data.corr()
 
# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

from sklearn.preprocessing import StandardScaler
features = ['TV', 'Radio', 'Newspaper']
# Separating out the features
x = data.loc[:, features].values
# Separating out the target
y = data.loc[:,['Sales']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

corrmat = principalDf.corr()
 
# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# Instantiate and fit our model.
regr = linear_model.LinearRegression()
Y = data['Sales'].values.reshape(-1, 1)
X = principalDf[['principal component 1','principal component 2']]
regr.fit(X, Y)

# Extract predicted values.
predicted2 = regr.predict(X).ravel()
actual2 = data['Sales']

# Calculate the error, aka residual.
residual2 = actual2 - predicted2

plt.hist(residual2)
plt.title('Residual counts')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.show()



