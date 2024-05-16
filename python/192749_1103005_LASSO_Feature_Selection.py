import graphlab

sales = graphlab.SFrame('kc_house_data.gl/')

from math import log, sqrt
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']

# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to float, before creating a new feature.
sales['floors'] = sales['floors'].astype(float) 
sales['floors_square'] = sales['floors']*sales['floors']

sales.head()

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = graphlab.linear_regression.create(
    sales,
    target='price',
    features=all_features,
    validation_set=None,
    verbose = False,
    l2_penalty=0.,
    l1_penalty=1e10)

model_all['coefficients'][model_all['coefficients']['value'] != 0.0][:]

(training_and_validation, testing) = sales.random_split(.9,seed=1) # initial train/test split
(training, validation) = training_and_validation.random_split(0.5, seed=1) # split training into train and validate

import numpy as np

validation_rss = {}
for l1_penalty in np.logspace(1, 7, num=13):
    model = graphlab.linear_regression.create(
        training,
        target='price',
        features=all_features,
        validation_set=None,
        verbose = False,
        l2_penalty=0.,
        l1_penalty=l1_penalty)

    predictions = model.predict(validation)
    residuals = validation['price'] - predictions
    rss = (residuals*residuals).sum()

    validation_rss[l1_penalty] = rss

print min(validation_rss.items(), key=lambda x: x[1])

min(validation_rss.items(), key=lambda x: x[1])[0]

model = graphlab.linear_regression.create(
        training,
        target='price',
        features=all_features,
        validation_set=None,
        verbose = False,
        l2_penalty=0.,
        l1_penalty=10.0)

len(model['coefficients'][model['coefficients']['value'] != 0.0])

max_nonzeros = 7

l1_penalty_values = np.logspace(8, 10, num=20)

coef_dict = {}
for l1_penalty in l1_penalty_values:
    model = graphlab.linear_regression.create(
        training,
        target ='price',
        features=all_features,
        validation_set=None,
        verbose=None,
        l2_penalty=0.,
        l1_penalty=l1_penalty)

    coef_dict[l1_penalty] = model['coefficients']['value'].nnz()

coef_dict

l1_penalty_min = -1e+99
for l1_penalty, non_zeros in coef_dict.items():
    if non_zeros <= max_nonzeros:
        continue
    
    l1_penalty_min = max(l1_penalty_min, l1_penalty)
    
l1_penalty_min

l1_penalty_max = 1e+99
for l1_penalty, non_zeros in coef_dict.items():
    if non_zeros >= max_nonzeros:
        continue
    
    l1_penalty_max = min(l1_penalty_max, l1_penalty)
    
l1_penalty_max

l1_penalty_min

l1_penalty_values = np.linspace(l1_penalty_min, l1_penalty_max, 20)

validation_rss = {}
for l1_penalty in l1_penalty_values:
    model = graphlab.linear_regression.create(
        training, target='price',
        features=all_features,
        validation_set=None,
        verbose = False,
        l2_penalty=0.,
        l1_penalty=l1_penalty)

    predictions = model.predict(validation)
    residuals = validation['price'] - predictions
    rss = (residuals*residuals).sum()

    validation_rss[l1_penalty] = rss, model['coefficients']['value'].nnz()

validation_rss

best_rss = 1e+99
for l1_penalty, (rss, non_zeros) in validation_rss.items():    
    if (non_zeros == max_nonzeros) and (l1_penalty < best_rss):
        best_rss = rss
        best_l1_penalty = l1_penalty
        
print best_rss, best_l1_penalty

best_l1_penalty

model = graphlab.linear_regression.create(
    training,
    target='price',
    features=all_features,
    validation_set=None,
    verbose = False,
    l2_penalty=0.,
    l1_penalty=best_l1_penalty)

print model["coefficients"][model["coefficients"]["value"] != 0.0][:]

