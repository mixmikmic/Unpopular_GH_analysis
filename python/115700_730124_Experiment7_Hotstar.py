get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import scipy as sp
import time
import gc

import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

sns.set_style('dark')

SEED = 31314
np.random.seed(SEED)

import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('run ../src/data/HotstarDataset.py')
get_ipython().magic('run ../src/features/categorical_features.py')
get_ipython().magic('run ../src/features/util.py')
get_ipython().magic('run ../src/models/cross_validation.py')

dataset = Hotstar('../data/raw/5f828822-4--4-hotstar_dataset/')
dataset.load_data('../data/processed/hotstar_processed.feather')

data_processed = dataset.data
train_mask     = dataset.get_train_mask() 

# replace cricket, football, badminton, hocket with sports
data_processed['genres'] = data_processed.genres                                        .str                                        .replace('Cricket|Football|Badminton|Hockey|Volleyball|Swimming|Table Tennis|Tennis|Athletics|Boxing|Formula1|FormulaE|IndiaVsSa|Kabaddi', 'Sport')

# ohe genres
genres_ohe_encoded = encode_ohe(data_processed.genres)

# count based features

data_processed['num_cities'] = count_feature(data_processed.cities)
data_processed['num_genres'] = count_feature(data_processed.genres)
data_processed['num_titles'] = count_feature(data_processed.titles)
data_processed['num_tod']    = count_feature(data_processed.tod)
data_processed['num_dow']    = count_feature(data_processed.dow)

# watch time by genres
data_processed['watch_time_sec'] = num_seconds_watched(data_processed.genres)

features = pd.concat((data_processed[['num_cities', 'num_genres',
                'num_titles', 'num_tod',
                'num_dow', 'watch_time_sec',
                'segment'
               ]], genres_ohe_encoded), axis='columns')

save_file(features, '../data/processed/hotstar_processed_exp_7.feather')

features.columns

features.columns

X = features.loc[train_mask, features.columns.drop('segment')]
y = features.loc[train_mask, 'segment']
Xtest = features.loc[~train_mask, features.columns.drop('segment')]

params = {
    'stratify': y,
    'test_size': .3,
    'random_state': SEED
}

X_train, X_test, y_train, y_test = get_train_test_split(X, y, **params)

# further split train set into train and validation set
params = {
    'stratify': y_train,
    'test_size': .2,
    'random_state': SEED
}

Xtr, Xte, ytr, yte = get_train_test_split(X_train, y_train, **params)

dtrain = xgb.DMatrix(Xtr, ytr, missing=np.nan, feature_names=features.columns.drop('segment'))
dval   = xgb.DMatrix(Xte, yte, missing=np.nan, feature_names=features.columns.drop('segment'))

xgb_params = {
    'eta': 0.1,
    'max_depth': 5,
    'gamma': 1,
    'colsample_bytree': .7,
    'min_child_weight': 3.,
    'subsample': 1.,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': SEED,
    'silent': 1
}

n_estimators = 500

watchlist = [(dtrain, 'train'), (dval, 'val')]

model = xgb.train(xgb_params, dtrain, num_boost_round=n_estimators, verbose_eval=10,
                  evals=watchlist
                 )









