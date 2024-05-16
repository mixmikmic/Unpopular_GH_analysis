import numpy as np
import pandas as pd
cars = pd.read_csv("kuiper.csv")
cars.head()

cars.describe()

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['font.size'] = 16

plt.plot(cars['Mileage'], cars['Price'], 'wo')
plt.xlabel('Mileage')
plt.ylabel('Price')

import statsmodels.api as sm
X = sm.add_constant(cars['Mileage'])
regmodel = sm.OLS(cars['Price'], X, missing='none')
result = regmodel.fit()
print result.summary()

intercept = result.params[0]
slope = result.params[1]
lines = plt.plot(cars['Mileage'], cars['Price'], 'wo', cars['Mileage'], slope * cars['Mileage'] + intercept, 'k-')
plt.xlabel('Mileage')
plt.ylabel('Price')

plt.plot(result.resid, 'wo')
plt.xlabel('Index')
plt.ylabel('Residual')

n, bins, patches = plt.hist(result.resid / 1000)
plt.xlabel('Residual / 1000')
plt.ylabel('Count')

cars_wo_cruise = cars[cars['Cruise'] == 0]
cars_w_cruise = cars[cars['Cruise'] == 1]
plt.plot(cars_wo_cruise['Mileage'], cars_wo_cruise['Price'], 'ko')
plt.plot(cars_w_cruise['Mileage'], cars_w_cruise['Price'], 'wo', alpha=0.5)
plt.xlabel('Mileage')
plt.ylabel('Price')

X = sm.add_constant(cars_wo_cruise['Mileage'])
regmodel = sm.OLS(cars_wo_cruise['Price'], X, missing='none')
print regmodel.fit().summary()

X = sm.add_constant(cars_w_cruise['Mileage'])
regmodel = sm.OLS(cars_w_cruise['Price'], X, missing='none')
print regmodel.fit().summary()

import statsmodels.formula.api as smf
result = smf.ols(formula='Price ~ Mileage + Cruise', data=cars).fit()
print result.summary()

intercept, mileage, cruise = result.params
x = np.linspace(min(cars['Mileage']), max(cars['Mileage']))
plt.plot(cars['Mileage'], cars['Price'], 'wo',          x, mileage * x + cruise * 0 + intercept, 'k-',          x, mileage * x + cruise * 1 + intercept, 'r-')
plt.xlabel('Mileage')
plt.ylabel('Price')

result = smf.ols(formula='Price ~ Mileage + Cruise + Mileage:Cruise', data=cars).fit()
print result.summary()

cars['Type'].describe()

print cars['Type'].count()

result = smf.ols(formula='Price ~ Mileage + Type', data=cars).fit()
print result.summary()

result = smf.ols(formula='Price ~ Mileage + I(Mileage**2)', data=cars).fit()
print result.summary()

intercept, mileage, mileage2 = result.params
x = np.linspace(min(cars['Mileage']), max(cars['Mileage']))
plt.plot(cars['Mileage'], cars['Price'], 'wo',          x, mileage * x + mileage2 * x * x + intercept, 'k-')
plt.xlabel('Mileage')
plt.ylabel('Price')

