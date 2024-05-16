get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import scipy as sp

import gc
import json
import time

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

from bayes_opt import BayesianOptimization

sns.set_style('dark')

SEED = 2123
np.random.seed(SEED)

import warnings
warnings.filterwarnings('ignore')

get_ipython().magic('run ../src/models/cross_validation.py')

with open('../data/raw/5f828822-4--4-hotstar_dataset/train_data.json', 'r') as infile:
    train_json = json.load(infile)
    train      = pd.DataFrame.from_dict(train_json, orient='index')
    
    train.reset_index(level=0, inplace=True)
    train.rename(columns = {'index':'ID'},inplace=True)
    
    infile.close()
    
with open('../data/raw/5f828822-4--4-hotstar_dataset/test_data.json') as infile:
    test_json = json.load(infile)
    
    test = pd.DataFrame.from_dict(test_json, orient='index')
    test.reset_index(level=0, inplace=True)
    test.rename(columns = {'index':'ID'},inplace=True)
    
    infile.close()

# encode segment variable
lbl = LabelEncoder()
lbl.fit(train['segment'])

train['segment'] = lbl.transform(train['segment'])

data       = pd.concat((train, test))
train_mask = data.segment.notnull()

del train, test
gc.collect()

data.loc[train_mask, 'segment'].value_counts(normalize=True)

genre_dict_train = data.loc[train_mask, 'genres'].map(lambda x: x.split(','))                     .map(lambda x: dict((k.strip(), int(v.strip())) for k,v in 
                                          (item.split(':') for item in x)))

genre_dict_test  = data.loc[~train_mask, 'genres'].map(lambda x: x.split(','))                     .map(lambda x: dict((k.strip(), int(v.strip())) for k,v in 
                                          (item.split(':') for item in x)))
    
dv    = DictVectorizer(sparse=False)
X     = dv.fit_transform(genre_dict_train)
Xtest = dv.transform(genre_dict_test)

y     = data.loc[train_mask, 'segment']

# convert it into pandas dataframe
X = pd.DataFrame(X)
y = pd.Series(y)

Xtest = pd.DataFrame(Xtest)

params = {
    'stratify': y,
    'test_size': .3,
    'random_state': SEED
}

X_train, X_test, y_train, y_test = get_train_test_split(X, y, **params)

rf = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state=SEED)

auc_scores = cross_validation(X_train, y_train, rf, 'auc', SEED)

print('Mean AUC score: {0} and std: {1}'.format(np.mean(auc_scores), np.std(auc_scores)))

def rfccv(n_estimators, min_samples_split, max_depth):
    skf = StratifiedKFold(n_splits=3, random_state=SEED)
    val = cross_val_score(
        RandomForestClassifier(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_depth=int(max_depth),
                               random_state=SEED
                              ),
        X_train, y_train, scoring='roc_auc', cv=skf
    ).mean()
    
    return val

def logccv(C):
    skf = StratifiedKFold(n_splits=3, random_state=SEED)
    
    val = cross_val_score(
        LogisticRegression(C=C,
        n_jobs=2,
        class_weight='balanced',
        random_state=SEED
                          ),
        X_train, y_train, scoring='roc_auc', cv=skf
    ).mean()
    
    return val

def parameter_search(rf):
    gp_params = {
        'alpha': 1e-5
    }
    
    if rf:
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
        
    else:
        logcBO = BayesianOptimization(
            logccv,
            {
                'C': (.01, 100)
            }
        )
        
        logcBO.maximize(n_iter=10, **gp_params)
        print('Logistic Regression: %f' % logcBO.res['max']['max_val'])

start = time.time()
parameter_search()
end   = time.time()

print('Took: {} seconds to do parameter tuning'.format(end - start))

start = time.time()
parameter_search(rf=False)
end   = time.time()

print('Took: {} seconds to do parameter tuning'.format(end - start))

def test_model(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    print('Log Loss on test set: {}'.format(roc_auc_score(y_test, preds)))

rf = RandomForestClassifier(n_estimators=219, 
                                max_depth=11, 
                                min_samples_split=19, 
                                random_state=SEED)
    
test_model(X_train, y_train, X_test, y_test, rf)

log = LogisticRegression(C=.01, class_weight='balanced', random_state=SEED)
    
test_model(X_train, y_train, X_test, y_test, log)

def full_training(X, y, Xtest, model, model_name, save=True):
    model.fit(X, y)
    final_preds = model.predict_proba(Xtest)[:, 1]
    
    if save:
        joblib.dump(model, '../models/%s'%(model_name))
        
    return final_preds

log = LogisticRegression(C=.01, class_weight='balanced', random_state=SEED)

final_preds = full_training(X, y, Xtest, log, 'log_genre_wt.pkl')

sub = pd.read_csv('../data/raw/5f828822-4--4-hotstar_dataset/sample_submission.csv')

sub['segment'] = final_preds
sub.to_csv('../submissions/hotstar/log_genre_watch_times.csv', index=False)

