import random
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

# exact slope and intercept
m = 2.25
b = 3.75

# generate points on the line with Gaussian noise
points = 100
errors = [random.normalvariate(0, 0.5) for _ in range(points)]
x = [random.random() for _ in range(points)]
y = [m * x_ + b + e for x_, e in zip(x, errors)]

plt.plot(x, y, 'wo')
plt.xlabel('X')
plt.ylabel('y')

from sklearn import linear_model
linreg = linear_model.LinearRegression()
X = [[x_] for x_ in x]
linreg.fit(X, y)

print linreg.coef_[0], linreg.intercept_

plt.plot(x, y, 'wo')
plt.plot([0, 1], [linreg.intercept_, linreg.coef_[0] * 1.0 + linreg.intercept_], 'k-')
plt.xlabel('X')
plt.ylabel('y')

