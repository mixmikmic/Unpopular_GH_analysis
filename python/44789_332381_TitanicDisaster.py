import numpy as np
import pandas as pd
import sklearn.linear_model as lm
#from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

train = pd.read_csv('http://localhost:8888/files/input/train.csv')
test = pd.read_csv('http://localhost:8888/files/input/test.csv')

print ("Dimension of train data {}".format(train.shape))
print ("Dimension of test data{}".format(test.shape))

print ("Basic statistical description:")
train.describe()

train.tail()

test.head()

## Plotting data here 
## https://www.kaggle.com/amanullahtariq/titanic/exploratory-tutorial-titanic-disaster/editnb

plt.rc('font', size=13)
fig = plt.figure(figsize=(18, 8))
fig.show()

