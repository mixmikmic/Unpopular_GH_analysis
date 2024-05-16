carat, cut, clarity = 1.5, 3, 5
price = -5269 + 8413 * carat + 158.1 * cut + 454 * clarity
print("The price is ", price)

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
data = pd.read_csv("diamonds.csv")
test = pd.read_csv("new-diamonds.csv")

data.head()

data.median()

test.info()

test['predict'] = -5269 + 8413 * test['carat'] + 158.1 * test['cut_ord'] + 454.0 * test['clarity_ord']

plot1, = plt.plot(data["carat"],data["price"],'.r',label ="known price")
plot2, = plt.plot(test["carat"],test["predict"],'.b',label = "predicted price")
plt.legend(handles=[plot1, plot2],fontsize = 15)
plt.xlabel("carat",fontsize = 20)
plt.ylabel("price",fontsize = 20)
plt.show()

test[test['predict']>19000]

total = sum(test['predict'])*0.8
print("The bid price for 3000 diamonds:", total)

