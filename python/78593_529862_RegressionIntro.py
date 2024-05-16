# import relevant modules
import pandas as pd
import numpy as np
import quandl, math

# Machine Learning
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

#Visualization
import matplotlib
get_ipython().magic('matplotlib inline')
matplotlib.style.use('ggplot')

# Get unique quandl key by creating a free account with quandl 
# And directly load financial data from GOOGL

quandl.ApiConfig.api_key = 'q-UWpMLYsWKFejy5y-4a'
df = quandl.get('WIKI/GOOGL')

# Getting a peek into data
print(df.columns)
print(df.head(2))

# Discarding features that aren't useful
df = df[['Adj. Open','Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
print(df.head(2))

# define a new feature, HL_PCT
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/(df['Adj. Low']*100)

# define a new feature percentage change
df['PCT_CHNG'] = (df['Adj. Close'] - df['Adj. Open'])/(df['Adj. Open']*100)

df = df[['Adj. Close', 'HL_PCT', 'PCT_CHNG', 'Adj. Volume']]

print(df.head(3))

# Check which columns have missing data
for column in df.columns:
    if np.any(pd.isnull(df[column])) == True:
        print(column)

# pick a forecast column
forecast_col = 'Adj. Close'

# Plot features

df.plot( x = 'HL_PCT', y = 'PCT_CHNG', style = 'o')

# Chosing 1% of total days as forecast, so length of forecast data is 0.01*length
print('length = ',len(df))
forecast_out = math.ceil(0.01*len(df))

# Creating label and shifting data as per 'forecast_out'
df['label'] = df[forecast_col].shift(-forecast_out)
print(df.head(2))

# If we look at the tail, it consists of forecast_out rows with NAN in Label column 
print(df.tail(2))
print('\n')
# We can simply drop those rows
df.dropna(inplace=True)
print(df.tail(2))

# Define features (X) and Label (y)
# For X drop label and index
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
print('X[1,:] = ', X[1,:])
print('y[1] = ',y[1])
print('length of X and y: ', len(X), len(y))

# Use skalearn, preposessing to scale features
X = preprocessing.scale(X)
print(X[1,:])

# Cross validation (split into test and train data)
# test_size = 0.2 ==> 20% data is test data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

print('length of X_train and x_test: ', len(X_train), len(X_test))

# Train
clf = LinearRegression()
clf.fit(X_train,y_train)
# Test
accuracy = clf.score(X_test, y_test)
print("Accuracy of Linear Regression: ", accuracy)

# Train
clf2 = svm.SVR()
clf2.fit(X_train,y_train)
# Test
accuracy = clf2.score(X_test, y_test)
print("Accuracy of SVM: ", accuracy)

