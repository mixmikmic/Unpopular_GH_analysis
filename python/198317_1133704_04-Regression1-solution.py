import pandas as pd
data = pd.read_csv('../data/bikes.csv', sep=',')
data.head()

from sklearn.model_selection import train_test_split

X_all = data[['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit','temp', 'atemp', 'hum', 'windspeed']]
y_all = data['cnt']


X_train, X_test, y_train, y_test = train_test_split(
    X_all, 
    y_all,
    random_state=1,
    test_size=0.2)

print('Train size: {}'.format(len(X_train)))
print('Test size: {}'.format(len(X_test)))

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

regr = LinearRegression()

#regr = Pipeline([('std', StandardScaler()),
#                 ('svr', svm.SVR(kernel='linear'))])

#regr = GradientBoostingRegressor(n_estimators=100, max_depth=4)

regr.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

y_pred = regr.predict(X_test)
print ("Test mean: {}, std: {}".format(np.mean(y_test), np.std(y_test)))
print("Test Root mean squared error: {:.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
print("Test Mean absolute error: {:.2f}".format(mean_absolute_error(y_test, y_pred)))

y_pred = regr.predict(X_train)
print("Train Root mean squared error: %.2f"
      % np.sqrt(mean_squared_error(y_train, y_pred)))
print("Train Mean absolute error: %.2f"
      % mean_absolute_error(y_train, y_pred))

print('Coefficients: \n', regr.coef_)

