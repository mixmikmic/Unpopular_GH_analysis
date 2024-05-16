get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import gc
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

from bayes_opt import BayesianOptimization

pd.set_option('max_columns', None)

sns.set_style('dark')

SEED = 213123
np.random.seed(SEED)

import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('run ../src/data/make_dataset.py')
get_ipython().magic('run ../src/models/cross_validation.py')

dataset = Dataset('../data/raw/4b699168-4-here_dataset/')

dataset.load_files()       .encode_target()       .rename_target()       .concat_data()       .save_data('../data/processed/processed.feather')

data       = dataset.data
train_mask = dataset.get_train_mask() 

features = ['AngleOfSign']
label    = 'Target'

X = data.loc[train_mask, features]
y = data.loc[train_mask, label]

Xtest = data.loc[~train_mask, features]

params = {
    'stratify': y,
    'test_size': .3,
    'random_state': SEED
}

X_train, X_test, y_train, y_test = get_train_test_split(X, y, **params)

y_train.value_counts(normalize=True)

rf = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=SEED)
ll_scores = cross_validation(X_train, y_train, rf, SEED)

print('Mean ll score: {0} and std: {1}'.format(np.mean(ll_scores), np.std(ll_scores)))

def rfccv(n_estimators, min_samples_split, max_depth):
    skf = StratifiedKFold(n_splits=3, random_state=SEED)
    val = cross_val_score(
        RandomForestClassifier(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_depth=int(max_depth),
                               random_state=SEED
                              ),
        X_train, y_train, scoring='neg_log_loss', cv=skf
    ).mean()
    
    return val

def parameter_search():
    gp_params = {
        'alpha': 1e-5
    }
    
    rfcBO = BayesianOptimization(
        rfccv,
        {
            'n_estimators': (10, 250),
            'min_samples_split': (2, 25),
            'max_depth': (5, 30)
        }
    )
    
    rfcBO.maximize(n_iter=10, **gp_params)
    print('RFC: %f' % rfcBO.res['max']['max_val'])

parameter_search()

def test_model(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=250, 
                                max_depth=5, 
                                min_samples_split=25, 
                                random_state=SEED)
    
    rf.fit(X_train, y_train)
    preds = rf.predict_proba(X_test)
    print('Log Loss on test set: {}'.format(log_loss(y_test, preds)))

test_model(X_train, y_train, X_test, y_test)

def full_training(X, y, Xtest, save=True):
    rf = RandomForestClassifier(n_estimators=250, 
                                max_depth=5, 
                                min_samples_split=25, 
                                random_state=SEED)
    
    rf.fit(X, y)
    final_preds = rf.predict_proba(Xtest)
    
    if save:
        joblib.dump(rf, '../models/rf_model_angle_of_sign.pkl')
        
    return final_preds

final_preds = full_training(X, y, Xtest)

data.loc[~train_mask, :].head(2)

sample_sub = dataset.sub
sample_sub.loc[:, ['Front', 'Left', 'Rear', 'Right']] = final_preds

sample_sub.to_csv('../submissions/predict_sign/rf_angle_of_sign.csv', index=False)

