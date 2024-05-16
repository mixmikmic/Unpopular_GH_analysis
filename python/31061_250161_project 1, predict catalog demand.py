import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
customer = pd.read_excel("p1-customers.xlsx")
mail = pd.read_excel("p1-mailinglist.xlsx")

customer.head()

#mail.head()

customer.info()

mail.info()

customer['Responded_to_Last_Catalog'].value_counts()

sns.boxplot(x='Responded_to_Last_Catalog', y = "Avg_Sale_Amount",data = customer)

sns.regplot(x='Avg_Num_Products_Purchased', y = "Avg_Sale_Amount",data = customer)

sns.boxplot(x="Customer_Segment", y = "Avg_Sale_Amount",data = customer)

sns.boxplot(x="City", y = "Avg_Sale_Amount",data = customer)

sns.jointplot(x='#_Years_as_Customer', y ='Avg_Sale_Amount', data = customer)

dummies = pd.get_dummies(customer['Customer_Segment'])
X = pd.concat([customer['Avg_Num_Products_Purchased'], dummies], axis=1)
X = X.drop("Credit Card Only",axis=1)  # use "Credit Card Only" as default
y = customer["Avg_Sale_Amount"]
X.head()

import statsmodels.api as sm
#from scipy import stats
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

from sklearn.linear_model import LinearRegression, Ridge
#model = LinearRegression() # ridicularly larege intercept
model = Ridge()
model.fit(X, y)
model.coef_, model.intercept_

model.score(X,y)

dummies = pd.get_dummies(mail['Customer_Segment'])
X_test = pd.concat([mail['Avg_Num_Products_Purchased'], dummies], axis=1)
X_test = X_test.drop("Credit Card Only", axis = 1)
X_test.head()

y_test = model.predict(X_test)

import numpy as np
print(np.median(y_test))
print(mail["Score_Yes"].median())

sum(mail["Score_Yes"]*y_test*0.5) - 6.5 *250

(mail["Score_Yes"]*y_test)[0]  # first record of revenue

