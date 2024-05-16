import numpy as np
import pandas as pd

data = pd.read_csv('Advertising.csv', index_col=0)
data.head()

data.describe()

data.corr()

import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=4, kind='reg')

import statsmodels.formula.api as smf
result = smf.ols(formula='Sales ~ TV', data=data[['TV', 'Sales']]).fit()
print result.summary()

residuals = data.Sales - (7.0326 + 0.0475 * data.TV)
plt.scatter(data.TV, residuals)

plt.hist(residuals)

from scipy.stats import anderson
a2, crit, sig = anderson(residuals, 'norm')
a2, crit, sig

result = smf.ols(formula='Sales ~ Radio', data=data[['Radio', 'Sales']]).fit()
print result.summary()

result = smf.ols(formula='Sales ~ Newspaper', data=data[['Newspaper', 'Sales']]).fit()
print result.summary()

result = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()
print result.summary()

result = smf.ols(formula='Sales ~ TV + Radio', data=data).fit()
print result.summary()

X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

from sklearn import metrics
print metrics.mean_absolute_error(y_test, y_pred)
print metrics.mean_squared_error(y_test, y_pred)
print np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print metrics.r2_score(y_test, y_pred)

X = X.drop('Newspaper', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

linreg.fit(X_train, y_train)
y_predict = linreg.predict(X_test)

print np.sqrt(metrics.mean_squared_error(y_test, y_predict))
print metrics.r2_score(y_test, y_predict)

def func(x_, y_):
    return linreg.intercept_ + x_ * linreg.coef_[0] + y_ * linreg.coef_[1]

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data.TV, data.Radio, data.Sales, color='r')

x = np.linspace(0, 300)
y = np.linspace(0, 50)
(X, Y) = np.meshgrid(x, y)
z = np.array([func(x,y) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = z.reshape(X.shape)
ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4, color='k', alpha=0.5)

ax.view_init(elev=20, azim=300)
ax.set_xlabel('TV')
ax.set_ylabel('Radio')
ax.set_zlabel('Sales')

