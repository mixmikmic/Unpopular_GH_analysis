import pandas as pd
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv('ENB2012_data.csv', na_filter=False)
df = df.drop(['Unnamed: 10','Unnamed: 11'], axis=1)
df['X1'] = pd.to_numeric(df['X1'], errors='coerce')
df['X2'] = pd.to_numeric(df['X2'], errors='coerce')
df['X3'] = pd.to_numeric(df['X3'], errors='coerce')
df['X4'] = pd.to_numeric(df['X4'], errors='coerce')
df['X5'] = pd.to_numeric(df['X5'], errors='coerce')
df['X6'] = pd.to_numeric(df['X6'], errors='coerce')
df['X7'] = pd.to_numeric(df['X7'], errors='coerce')
df['X8'] = pd.to_numeric(df['X8'], errors='coerce')
df['Y1'] = pd.to_numeric(df['Y1'], errors='coerce')
df['Y2'] = pd.to_numeric(df['Y2'], errors='coerce')

df = df.dropna()
print (df.dtypes)
print (df.head())
plt.show()
plt.plot(df.values[:,8])
plt.show()
plt.plot(df.values[:,9])
plt.close()

plt.scatter(df['Y1'], df['Y2'])
plt.show()
plt.close()

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import linear_model

train, test = train_test_split(df, test_size = 0.3)
X_tr = train.drop(['Y1','Y2'], axis=1)
y_tr = train['Y1']
test = test.sort_values('Y1')
X_te = test.drop(['Y1','Y2'], axis=1)
y_te = test['Y1']

reg_svr = svm.SVR()
reg_svr.fit(X_tr, y_tr)

reg_lin = linear_model.LinearRegression()
reg_lin.fit(X_tr, y_tr)

y_pre_svr = reg_svr.predict(X_te)
y_lin_svr = reg_lin.predict(X_te)
print ("Coefficient R^2 of the SVR prediction: " + str(reg_svr.score(X_tr, y_tr)))
print ("Coefficient R^2 of the Linear Regression prediction:" + str(reg_lin.score(X_tr, y_tr)))


plt.plot(y_pre_svr, label="Prediction for SVR")
plt.plot(y_te.values, label="Heating Load")
plt.plot(y_lin_svr, label="Prediction for linear")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

train, test = train_test_split(df, test_size = 0.3)
X_tr = train.drop(['Y1','Y2'], axis=1)
y_tr = train['Y2']
test = test.sort_values('Y2')
X_te = test.drop(['Y1','Y2'], axis=1)
y_te = test['Y2']

reg_svr = svm.SVR()
reg_svr.fit(X_tr, y_tr)

reg_lin = linear_model.LinearRegression()
reg_lin.fit(X_tr, y_tr)

y_pre_svr = reg_svr.predict(X_te)
y_lin_svr = reg_lin.predict(X_te)
print ("Coefficient R^2 of the SVR prediction: " + str(reg_svr.score(X_tr, y_tr)))
print ("Coefficient R^2 of the Linear Regression prediction: " + str(reg_lin.score(X_tr, y_tr)))

plt.plot(y_pre_svr, label="Prediction for SVR")
plt.plot(y_te.values, label="Cooling Load")
plt.plot(y_lin_svr, label="Prediction for linear")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()

# coefficients of linear model
print (reg_lin.coef_)



