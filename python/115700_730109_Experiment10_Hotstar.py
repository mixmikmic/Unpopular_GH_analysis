get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import scipy as sp

SEED = 1231
np.random.seed(SEED)

import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib

import xgboost as xgb

from itertools import combinations

sns.set_style('dark')

import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('run ../src/data/HotstarDataset.py')
get_ipython().magic('run ../src/features/categorical_features.py')
get_ipython().magic('run ../src/features/util.py')
get_ipython().magic('run ../src/models/cross_validation.py')
get_ipython().magic('run ../src/models/feature_selection.py')

# load dataset
dataset = Hotstar('../data/raw/5f828822-4--4-hotstar_dataset/')
dataset.load_data('../data/processed/hotstar_processed.feather')

data_processed = dataset.data
train_mask     = dataset.get_train_mask() 

# preprocessing: replacement map
genre_replacement_map = {
    'Thriller': 'Crime',
    'Horror': 'Crime',
    'Action': 'Action',
    'Hockey': 'Sport',
    'Kabaddi': 'Sport',
    'Formula1': 'Sport',
    'FormulaE': 'Sport',
    'Tennis': 'Sport',
    'Athletics': 'Sport',
    'Table Tennis': 'Sport',
    'Volleyball': 'Sport',
    'Boxing': 'Sport',
    'Football': 'Sport',
    'NA': 'Sport',
    'Swimming': 'Sport',
    'IndiavsSa': 'Sport',
    'Wildlife': 'Travel',
    'Science': 'Travel',
    'Documentary': 'Travel'
}

def cluster_genres(genres):
    for replacement_key in genre_replacement_map.keys():
        to_replace = genre_replacement_map[replacement_key]
        genres     = genres.str.replace(r'%s'%(replacement_key), to_replace)
    
    return genres
            

start = time.time()
data_processed['genres'] = cluster_genres(data_processed.genres)
end   = time.time()

print('Took: {} seconds'.format(end - start))

start = time.time()
ohe_genres = encode_ohe(data_processed.genres)
end   = time.time()

print('Took: {} seconds'.format(end - start))

def group_data(dd, degree=2):
    new_data = []
    columns  = []
    
    for indices in combinations(dd.columns, degree):
        key = '_'.join(list(indices))
        columns.append(key)
        
        new_data.append(np.product(dd.loc[:, list(indices)].values, axis=1))
    
    new_data = np.array(new_data)
    return pd.DataFrame(new_data.T, columns=columns)

start = time.time()
feature_interaction = group_data(ohe_genres)
end   = time.time()

print('Took: {} seconds'.format(end - start))

# concat different data frames
# data = pd.concat((ohe_genres, feature_interaction, data_processed.segment), axis='columns')
data = np.hstack((ohe_genres.values, 
                  feature_interaction.values,
                  data_processed.segment.values.reshape(-1, 1)
                 ))

columns = ohe_genres.columns.tolist() + feature_interaction.columns.tolist() + ['segment']
data = pd.DataFrame(data, columns=columns)
save_file(data, '../data/processed/hotstar_processed_exp_10.feather')

del data_processed, ohe_genres, feature_interaction
gc.collect()

data = load_file('../data/processed/hotstar_processed_exp_10.feather')
train_mask = data.segment.notnull()

f = data.columns.drop('segment')

X = data.loc[train_mask, f]
y = data.loc[train_mask, 'segment']

Xtest  = data.loc[~train_mask, f]

params = {
    'stratify': y,
    'test_size': .3,
    'random_state': SEED
}

X_train, X_test, y_train, y_test = get_train_test_split(X, y, **params)

params = {
    'stratify': y_train,
    'test_size': .2,
    'random_state': SEED
}

Xtr, Xte, ytr, yte = get_train_test_split(X_train, y_train, **params)

# train a logistic regression model
model = LogisticRegression(C=.01, class_weight='balanced', random_state=SEED)
model.fit(Xtr, ytr)

preds = model.predict_proba(Xte)[:, 1]
print('AUC: {}'.format(roc_auc_score(yte, preds)))

# train a random forest model
model = RandomForestClassifier(n_estimators=100, max_depth=7,
                               max_features=.3, n_jobs=2, random_state=SEED)
model.fit(Xtr, ytr)

preds = model.predict_proba(Xte)[:, 1]
print('AUC: {}'.format(roc_auc_score(yte, preds)))

# train a extreme gradient boosting model
model = xgb.XGBClassifier(colsample_bytree=.6, seed=SEED)

model.fit(Xtr, ytr)

preds = model.predict_proba(Xte)[:, 1]
print('AUC: {}'.format(roc_auc_score(yte, preds)))

start = time.time()
model = LogisticRegression(random_state=SEED)
greedy_feature_search(Xtr.iloc[:1000], ytr.iloc[:1000], model)
end = time.time()

print('Took: {} seconds'.format(end - start))

selected_features = [4, 6, 9, 12, 16, 27, 40, 48, 55, 57, 77, 80, 89, 99, 100, 105, 112, 116, 118, 121, 129, 146, 147, 155, 157, 168, 170, 172, 174, 175, 181]

joblib.dump(selected_features, '../data/interim/experiment_10_selected_features.pkl')

model.fit(Xtr.iloc[:, selected_features], ytr)
preds = model.predict_proba(Xte.iloc[:, selected_features])[:, 1]

print('AUC: {}'.format(roc_auc_score(yte, preds)))

model.fit(X_train.iloc[:, selected_features], y_train)
preds = model.predict_proba(X_test.iloc[:, selected_features])[:, 1]

print('AUC: {}'.format(roc_auc_score(y_test, preds)))

start = time.time()
model = xgb.XGBClassifier(seed=SEED)
greedy_feature_search(Xtr.iloc[:1000], ytr.iloc[:1000], model)
end = time.time()

print('Took: {} seconds'.format(end - start))

selected_features = [4, 6, 14, 25, 48, 90, 107, 116, 129, 161, 163, 169, 177, 178]
joblib.dump(selected_features, '../data/interim/experiment_10_selected_features_xgboost.pkl')

model = xgb.XGBClassifier(n_estimators=150, max_depth=4, seed=SEED, learning_rate=.1)
model.fit(Xtr.iloc[:, selected_features], ytr)

preds = model.predict_proba(Xte.iloc[:, selected_features])[:, 1]
print('AUC: {}'.format(roc_auc_score(yte, preds)))

model.fit(X_train.iloc[:, selected_features], y_train)

preds = model.predict_proba(X_test.iloc[:, selected_features])[:, 1]
print('AUC: {}'.format(roc_auc_score(y_test, preds)))

# full training
model.fit(X.iloc[:, selected_features], y)
final_preds = model.predict_proba(Xtest.iloc[:, selected_features])[:, 1]

sub            = pd.read_csv('../data/raw/5f828822-4--4-hotstar_dataset/sample_submission.csv')
sub['segment'] = final_preds
sub['ID']      = data_processed.loc[~train_mask, 'ID'].values
sub.to_csv('../submissions/hotstar/xgb_experiment_10.csv', index=False)



