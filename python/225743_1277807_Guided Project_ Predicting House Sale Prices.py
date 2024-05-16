import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', 500)
data = pd.read_csv("AmesHousing.tsv", delimiter='\t')

print(data.shape)
print(len(str(data.shape))*'-')
print(data.dtypes.value_counts())
data.head()

#Create a new feature, 'years_to_sell'.
data['years_to_sell'] = data['Yr Sold'] - data['Year Remod/Add'] 
data = data[data['years_to_sell'] >= 0]

#Remove features that are not useful for machine learning.
data = data.drop(['Order', 'PID'], axis=1)

#Remove features that leak sales data.
data = data.drop(['Mo Sold', 'Yr Sold', 'Sale Type', 'Sale Condition'], axis=1)

#Drop columns with more than 5% missing values
is_null_counts = data.isnull().sum()
features_col = is_null_counts[is_null_counts < 2930*0.05].index


data = data[features_col]
data.head()

numerical_cols = data.dtypes[data.dtypes != 'object'].index
numerical_data = data[numerical_cols]

numerical_data = numerical_data.fillna(data.mode().iloc[0])
numerical_data.isnull().sum().sort_values(ascending = False)

num_corr = numerical_data.corr()['SalePrice'].abs().sort_values(ascending = False)
num_corr

num_corr = num_corr[num_corr > 0.4]
high_corr_cols = num_corr.index

hi_corr_numerical_data = numerical_data[high_corr_cols]

text_cols = data.dtypes[data.dtypes == 'object'].index
text_data = data[text_cols]

text_null_counts = text_data.isnull().sum()
text_not_null_cols = text_null_counts[text_null_counts < 1].index

text_data = text_data[text_not_null_cols]

nominal_cols = ['MS Zoning', 'Street', 'Alley', 'Land Contour', 'Lot Config', 'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style', 'Overall Qual', 'Roof Style', 'Roof Mat1', 'Exterior 1st',  'Exterior 2nd', 'Mas Vnr Type', 'Foundation', 'Heating', 'Central Air'] 
nominal_num_col = ['MS SubClass']

#Finds nominal columns in text_data
nominal_text_col = []
for col in nominal_cols:
    if col in text_data.columns:
        nominal_text_col.append(col)
nominal_text_col

text_data = text_data[nominal_text_col]

for col in nominal_text_col:
    print(col)
    print(text_data[col].value_counts())
    print("-"*10)

nominal_text_col_unique = []
for col in nominal_text_col:
    if len(text_data[col].value_counts()) <= 10:
        nominal_text_col_unique.append(col)
               
text_data = text_data[nominal_text_col_unique]

#Create dummy columns for nominal text columns, then create a dataframe.
for col in text_data.columns:
    text_data[col] = text_data[col].astype('category')   
categorical_text_data = pd.get_dummies(text_data)    
categorical_text_data.head()

#Create dummy columns for nominal numerical columns, then create a dataframe.
for col in numerical_data.columns:
    if col in nominal_num_col:
        numerical_data[col] = numerical_data[col].astype('category')  
              
categorical_numerical_data = pd.get_dummies(numerical_data.select_dtypes(include=['category'])) 

categorical_data = pd.concat([categorical_text_data, categorical_numerical_data], axis=1)

hi_corr_numerical_data.head()

categorical_data.head()

final_data = pd.concat([hi_corr_numerical_data, categorical_data], axis=1)

def transform_features(data, percent_missing=0.05):
    
    #Adding relevant features:
    data['years_since_remod'] = data['Year Built'] - data['Year Remod/Add']
    data['years_to_sell'] = data['Yr Sold'] - data['Year Built']
    data = data[data['years_since_remod'] >= 0]
    data = data[data['years_to_sell'] >= 0]
    
    #Remove columns not useful for machine learning
    data = data.drop(['Order', 'PID', 'Year Built', 'Year Remod/Add'], axis=1)
    
    #Remove columns that leaks sale data
    data = data.drop(['Mo Sold', 'Yr Sold', 'Sale Type', 'Sale Condition'], axis=1)
    
    #Drop columns with too many missing values defined by the function
    is_null_counts = data.isnull().sum()
    low_NaN_cols = is_null_counts[is_null_counts < len(data)*percent_missing].index
    
    transformed_data = data[low_NaN_cols]    
    return transformed_data

def select_features(data, corr_threshold=0.4, unique_threshold=10):  
    
    #Fill missing numerical columns with the mode.
    numerical_cols = data.dtypes[data.dtypes != 'object'].index
    numerical_data = data[numerical_cols]

    numerical_data = numerical_data.fillna(data.mode().iloc[0])
    numerical_data.isnull().sum().sort_values(ascending = False)

    #Drop text columns with more than 1 missing value.
    text_cols = data.dtypes[data.dtypes == 'object'].index
    text_data = data[text_cols]

    text_null_counts = text_data.isnull().sum()
    text_not_null_cols = text_null_counts[text_null_counts < 1].index

    text_data = text_data[text_not_null_cols]

    num_corr = numerical_data.corr()['SalePrice'].abs().sort_values(ascending = False)

    num_corr = num_corr[num_corr > corr_threshold]
    high_corr_cols = num_corr.index

    #Apply the correlation threshold parameter
    hi_corr_numerical_data = numerical_data[high_corr_cols]
    

    #Nominal columns from the documentation
    nominal_cols = ['MS Zoning', 'Street', 'Alley', 'Land Contour', 'Lot Config', 'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style', 'Overall Qual', 'Roof Style', 'Roof Mat1', 'Exterior 1st',  'Exterior 2nd', 'Mas Vnr Type', 'Foundation', 'Heating', 'Central Air'] 
    nominal_num_col = ['MS SubClass']

    #Finds nominal columns in text_data
    nominal_text_col = []
    for col in nominal_cols:
        if col in text_data.columns:
            nominal_text_col.append(col)
    nominal_text_col

    text_data = text_data[nominal_text_col]

    nominal_text_col_unique = []
    for col in nominal_text_col:
        if len(text_data[col].value_counts()) <= unique_threshold:
            nominal_text_col_unique.append(col)
        
        
    text_data = text_data[nominal_text_col_unique]
    text_data.head()

    #Set all these columns to categorical
    for col in text_data.columns:
        text_data[col] = text_data[col].astype('category')   
    categorical_text_data = pd.get_dummies(text_data)    

    #Change any nominal numerical columns to categorical, then returns a dataframe
    for col in numerical_data.columns:
        if col in nominal_num_col:
            numerical_data[col] = numerical_data[col].astype('category')  
           
    
    categorical_numerical_data = pd.get_dummies(numerical_data.select_dtypes(include=['category'])) 
    final_data = pd.concat([hi_corr_numerical_data, categorical_text_data, categorical_numerical_data], axis=1)

    return final_data

def train_and_test(data):

    train = data[0:1460]
    test = data[1460:]
    features = data.columns.drop(['SalePrice'])
    
    #train
    lr = LinearRegression()
    lr.fit(train[features], train['SalePrice'])
    #predict
    
    predictions = lr.predict(test[features])
    rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
    return rmse

data = pd.read_csv("AmesHousing.tsv", delimiter='\t')

transformed_data = transform_features(data, percent_missing=0.05)
final_data = select_features(transformed_data, 0.4, 10)
result = train_and_test(final_data)
result

from sklearn.model_selection import KFold

def train_and_test2(data, k=2):  
    rf = LinearRegression()
    if k == 0:
        train = data[0:1460]
        test = data[1460:]
        features = data.columns.drop(['SalePrice'])
    
        #train
        rf.fit(train[features], train['SalePrice'])
        
        #predict    
        predictions = rf.predict(test[features])
        rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
        return rmse
    
    elif k == 1:
        train = data[:1460]
        test = data[1460:]
        features = data.columns.drop(['SalePrice'])
        
        rf.fit(train[features], train["SalePrice"])
        predictions_one = rf.predict(test[features])        
        
        mse_one = mean_squared_error(test["SalePrice"], predictions_one)
        rmse_one = np.sqrt(mse_one)
        
        rf.fit(test[features], test["SalePrice"])
        predictions_two = rf.predict(train[features])        
       
        mse_two = mean_squared_error(train["SalePrice"], predictions_two)
        rmse_two = np.sqrt(mse_two)
        return np.mean([rmse_one, rmse_two])   
    
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state = 2)
        rmse_list = []
        for train_index, test_index in kf.split(data):
            train = data.iloc[train_index]
            test = data.iloc[test_index]
            features = data.columns.drop(['SalePrice'])
    
            #train
            rf.fit(train[features], train['SalePrice'])
        
            #predict    
            predictions = rf.predict(test[features])
        
            rmse = mean_squared_error(test['SalePrice'], predictions)**0.5
            rmse_list.append(rmse)
        return np.mean(rmse_list)

data = pd.read_csv("AmesHousing.tsv", delimiter='\t')

transformed_data = transform_features(data, percent_missing=0.05)
final_data = select_features(transformed_data, 0.4, 10)

results = []
for i in range(100):
    result = train_and_test2(final_data, k=i)
    results.append(result)
    
x = [i for i in range(100)]
y = results 
plt.plot(x, y)
plt.xlabel('Kfolds')
plt.ylabel('RMSE')

print(results[99])



