import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns
import numpy as np
sns.set()

df = pd.read_csv('finance.csv')
df.head()

print('layout A revenue:', ((df.layouta.iloc[1:].sum() - df.layouta.iloc[0]) / 12))
print('layout A ROI:', (((df.layouta.iloc[1:].sum() - df.layouta.iloc[0]) / 12) * 100) / df.layouta.iloc[0])

cost = df.layouta.iloc[0].copy()
copy_month = df.layouta.iloc[1:].values / 30.0
for i in range(df.layouta.iloc[1:].shape[0]):
    for k in range(30):
        cost = cost - copy_month[i]
        if cost <= 0:
            print('month:', i, 'day:',k)
            break

print('layout B revenue:', ((df.layoutb.iloc[1:].sum() - df.layoutb.iloc[0]) / 12))
print('layout B ROI:', (((df.layoutb.iloc[1:].sum() - df.layoutb.iloc[0]) / 12) * 100) / df.layoutb.iloc[0])

cost = df.layoutb.iloc[0].copy()
copy_month = df.layoutb.iloc[1:].values / 30.0
for i in range(df.layoutb.iloc[1:].shape[0]):
    for k in range(30):
        cost = cost - copy_month[i]
        if cost <= 0:
            print('month:', i, 'day:',k)
            break

print('layout C revenue:', ((df.layoutc.iloc[1:].sum() - df.layoutc.iloc[0]) / 12))
print('layout C ROI:', (((df.layoutc.iloc[1:].sum() - df.layoutc.iloc[0]) / 12) * 100) / df.layoutc.iloc[0])

cost = df.layoutc.iloc[0].copy()
copy_month = df.layoutc.iloc[1:].values / 30.0
for i in range(df.layoutc.iloc[1:].shape[0]):
    for k in range(30):
        cost = cost - copy_month[i]
        if cost <= 0:
            print('month:', i, 'day:',k)
            break

ratios = [0.1, 0.15, 0.2, 0.25]
yval = []
for i in ratios:
    yval.append(np.npv(i, df.layouta.iloc[1:]) * (1 + i))
    print('layout A NPV ', i * 100 , '%:', yval[-1])
    
regr = linear_model.LinearRegression().fit(np.array([ratios]).T, np.array([yval]).T)
y_pred = regr.predict(np.array([ratios]).T)
plt.scatter(ratios, yval, label = 'scatter')
plt.plot(ratios, y_pred[:, 0], c = 'g', label = 'linear line')
plt.legend()
plt.show()

ratios = [0.1, 0.15, 0.2, 0.25]
yval = []
for i in ratios:
    yval.append(np.npv(i, df.layoutb.iloc[1:]) * (1 + i))
    print('layout B NPV ', i * 100 , '%:', yval[-1])
    
regr = linear_model.LinearRegression().fit(np.array([ratios]).T, np.array([yval]).T)
y_pred = regr.predict(np.array([ratios]).T)
plt.scatter(ratios, yval, label = 'scatter')
plt.plot(ratios, y_pred[:, 0], c = 'g', label = 'linear line')
plt.legend()
plt.show()

ratios = [0.1, 0.15, 0.2, 0.25]
yval = []
for i in ratios:
    yval.append(np.npv(i, df.layoutc.iloc[1:]) * (1 + i))
    print('layout C NPV ', i * 100 , '%:', yval[-1])
    
regr = linear_model.LinearRegression().fit(np.array([ratios]).T, np.array([yval]).T)
y_pred = regr.predict(np.array([ratios]).T)
plt.scatter(ratios, yval, label = 'scatter')
plt.plot(ratios, y_pred[:, 0], c = 'g', label = 'linear line')
plt.legend()
plt.show()



