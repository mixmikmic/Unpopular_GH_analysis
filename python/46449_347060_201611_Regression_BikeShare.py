import pandas as pd

data = pd.read_csv("day.csv")
print data.columns # these are the columns available
print data[0:5] # these are the first 5 rows

from sklearn import linear_model
reg = linear_model.LinearRegression()

X = data[["temp", "workingday"]]
print X[0:5]

y = data["casual"]  # number of casual riders
print y[0:5]

reg.fit(X,y)

reg.coef_

reg.intercept_

reg.predict([[0.34, 0]])

(0.34 * 2146.13419876) + (0 * -809.03115377) + 338.38711661893797

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print X_train.shape
print X_test.shape
print
print y_train.shape
print y_test.shape

reg.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

y_pred = reg.predict(X_test)

# With 2 predictor variables
print mean_squared_error(y_test, y_pred)

