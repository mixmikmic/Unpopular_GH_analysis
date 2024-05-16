import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets # Import the linear regression function and dataset from scikit-learn
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error, r2_score

# Print figures in the notebook
get_ipython().magic('matplotlib inline')

boston = datasets.load_boston()

y = boston.target
X = boston.data
featureNames = boston.feature_names

print(boston.DESCR)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

plt.scatter(X_train[:,5], y_train)
plt.ylabel('Average Houseing Price')
plt.xlabel('Avearge Number of Rooms')

regr = linear_model.LinearRegression()
x_train = X_train[:,5][np.newaxis].T # regression expects a (#examples,#features) array shape
regr.fit(x_train, y_train)

plt.scatter(x_train, y_train)
plt.plot(x_train, regr.predict(x_train), c='r')
plt.ylabel('Average Houseing Price')
plt.xlabel('Avearge Number of Rooms')
plt.title('Regression Line on Training Data')

x_test = X_test[:,5][np.newaxis].T # regression expects a (#examples,#features) array shape
predictions = regr.predict(x_test)

plt.scatter(x_test, y_test)
plt.plot(x_test, predictions, c='r')
plt.ylabel('Average Houseing Price')
plt.xlabel('Avearge Number of Rooms')
plt.xlabel('Avearge Number of Rooms')
plt.title('Regression Line on Test Data')

mse = mean_squared_error(y_test, predictions)

print('The MSE is ' + '%.2f' % mse)

r2score = r2_score(y_test, predictions)

print('The R^2 score is ' + '%.2f' % r2score)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

predictions = regr.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print('The MSE is ' + '%.2f' % mse)

r2score = r2_score(y_test, predictions)
print('The R^2 score is ' + '%.2f' % r2score)



