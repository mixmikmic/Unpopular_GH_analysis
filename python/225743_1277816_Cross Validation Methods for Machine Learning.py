import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


data = pd.read_csv("AmesHousingFinal.csv")
print(data.shape)
data.head()

#93% of the data as training set
train = data[0:1460]
test = data[1460:]
features = data.columns.drop(['SalePrice'])

#train
lr = LinearRegression()
lr.fit(train[features], train['SalePrice'])

#predict
predictions = lr.predict(test[features])
rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
print('RMSE:')
print(rmse)

random_seeds = {}
for i in range(10):
    np.random.seed(i)
    randomed_index = np.random.permutation(data.index)
    randomed_df = data.reindex(randomed_index)

    train = randomed_df[0:1460]
    test = randomed_df[1460:]
    features = randomed_df.columns.drop(['SalePrice'])

    lr = LinearRegression()
    lr.fit(train[features], train['SalePrice'])

    predictions = lr.predict(test[features])
    rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
    random_seeds[i]=rmse
random_seeds

kf = KFold(n_splits=4, shuffle=True, random_state = 7)

rmse_list = []
for train_index, test_index in kf.split(data):
    train = data.iloc[train_index]
    test = data.iloc[test_index]
    features = data.columns.drop(['SalePrice'])
    
    #train
    lr.fit(train[features], train['SalePrice'])
        
    #predict    
    predictions = lr.predict(test[features])
        
    rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
    rmse_list.append(rmse)
print('RMSE from the four models:')
print(rmse_list)
print('----')
print('Average RMSE:')
print(np.mean(rmse_list))

kf = KFold(n_splits=len(data), shuffle=True, random_state = 7)
rmse_list = []

time_start = time.clock()
for train_index, test_index in kf.split(data):
    train = data.iloc[train_index]
    test = data.iloc[test_index]
    features = data.columns.drop(['SalePrice'])
    
    #train
    lr.fit(train[features], train['SalePrice'])
        
    #predict    
    predictions = lr.predict(test[features])
        
    rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
    rmse_list.append(rmse)    
time_stop = time.clock()

print('Processing time:')
print(str(time_stop-time_start) + ' seconds')
print('----')
print('Average RMSE:')
print(np.mean(rmse_list))

time_start = time.clock()

rmse_kfolds = []
for i in range(2, len(data),100):
    kf = KFold(n_splits=i, shuffle=True, random_state = 7)
    rmse_list = []
    for train_index, test_index in kf.split(data):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        features = data.columns.drop(['SalePrice'])
    
        #train
        lr.fit(train[features], train['SalePrice'])
        
        #predict    
        predictions = lr.predict(test[features])
        
        rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
        rmse_list.append(rmse)
    rmse_kfolds.append(np.mean(rmse_list))
time_stop = time.clock()

print('Processing time:')
print(str(time_stop-time_start) + ' seconds')


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

x = [i for i in range(2, len(data),100)]
y = rmse_kfolds 
plt.plot(x, y)
plt.xlabel('Kfolds')
plt.ylabel('Average RMSE')
plt.show()

#100% of the data as training set
train = data

#100% of the data as the test set
test = data

features = data.columns.drop(['SalePrice'])

#train
lr = LinearRegression()
lr.fit(train[features], train['SalePrice'])

#predict
predictions = lr.predict(test[features])
rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
print('RMSE:')
print(rmse)



