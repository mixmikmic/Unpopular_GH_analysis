## 1. Step 1: Identify the columns that could be useful ##
import pandas as pd

# Set index_col to False to avoid pandas thinking that the first column is row indexes (it's age).
income = pd.read_csv("income.csv", index_col=False)
columns_of_interest = ["age", "workclass", "education_num", "marital_status", "occupation",                       "relationship", "race", "sex", "hours_per_week", "native_country", 'high_income']

income = income.loc[:, columns_of_interest]
income.head(2)

## Step 2: Convert categorical columns to their numeric values ##

features_categorical = [ "workclass", "education_num", "marital_status", "occupation",                        "relationship", "race", "sex","native_country", "high_income"]
for c in features_categorical:
    income[c] = pd.Categorical(income[c]).codes

income.head(2)

## Step 3: Identify features and target ##

features = ["age", "workclass", "education_num", "marital_status", "occupation",            "relationship", "race", "sex", "hours_per_week", "native_country"]
target = 'high_income'

from sklearn.tree import DecisionTreeClassifier

# Instantiate the classifier
clf = DecisionTreeClassifier(random_state=1)

import numpy as np
import math

# Set a random seed so the shuffle is the same every time
np.random.seed(1)

# Shuffle the rows  
# This permutes the index randomly using numpy.random.permutation
# Then, it reindexes the dataframe with the result
# The net effect is to put the rows into random order
income = income.reindex(np.random.permutation(income.index))

# 80% to train and 20% to test
train_max_row = math.floor(income.shape[0] * .8)

train = income.iloc[:train_max_row, :]
test = income.iloc[train_max_row:, :]

# Fit the model

clf.fit(train[features], train[target])

# Making predictions
predictions = clf.predict(test[features])
predictions[:2]

from sklearn.metrics import roc_auc_score

test_auc = roc_auc_score(test[target], predictions)

print(test_auc)

train_predictions = clf.predict(train[columns])

train_auc = roc_auc_score(train[target], train_predictions)

print(train_auc)

def get_aucs(max_depth):
    # Decision trees model with max_depth 
    clf = DecisionTreeClassifier(random_state=1, max_depth=max_depth)

    clf.fit(train[columns], train[target])

    # Test AUC
    predictions = clf.predict(test[columns])
    test_auc = roc_auc_score(test[target], predictions)

    # Train AUC
    predictions_train = clf.predict(train[columns])
    train_auc = roc_auc_score(train[target], predictions_train)
    
    return test_auc, train_auc

depth_values = np.arange(2, 40)
auc_values = np.zeros((len(depth_values), 3))
for i, val in enumerate(depth_values):
    test_auc, train_auc = get_aucs(val)
    auc_values[i, 0]  = val
    auc_values[i, 1]  = test_auc
    auc_values[i, 2]  = train_auc

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})

plt.figure(figsize=(8,4))
plt.plot(auc_values[:,0], auc_values[:,1], label='Test AUC') 
plt.plot(auc_values[:,0], auc_values[:,2], color='b', label='Train AUC')
plt.legend()
plt.xlabel('Maximum Tree Depth')
plt.ylabel('AUC')

plt.show()

