get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt
import seaborn as sns

import time

sns.set_style('dark')

get_ipython().magic('run ../src/features/util.py')
get_ipython().magic('run ../src/models/cross_validation.py')

SEED = 1313141
np.random.seed(SEED)

# laod files
data = load_file('../data/processed/processed.feather')
train_mask = data.Target.notnull()

# encode detected camera
lbl = LabelEncoder()
data['DetectedCamera'] = lbl.fit_transform(data.DetectedCamera)

# add area
data['SignArea'] = np.log1p(data['SignHeight'] * data['SignWidth'])

sns.boxplot(x='Target', y='SignArea', data=data.loc[train_mask, :]);

sns.kdeplot(data.loc[train_mask & (data.Target == 0), 'SignArea'])
sns.kdeplot(data.loc[train_mask & (data.Target == 1), 'SignArea'])
sns.kdeplot(data.loc[train_mask & (data.Target == 2), 'SignArea'])
sns.kdeplot(data.loc[train_mask & (data.Target == 3), 'SignArea']);

sns.boxplot(x='Target', y='AngleOfSign', data=data.loc[train_mask, :])

sns.distplot(data.loc[train_mask & (data.Target == 0.0), 'AngleOfSign'])

sns.distplot(data.loc[train_mask & (data.Target == 1.0), 'AngleOfSign'])

sns.distplot(data.loc[train_mask & (data.Target == 2.0), 'AngleOfSign'])

sns.lmplot(x='AngleOfSign', y='SignAspectRatio', hue='Target', data=data.loc[train_mask & (data.Target.isin([0, 1])), :], 
           fit_reg=False);

sns.lmplot(x='AngleOfSign', y='SignArea', hue='Target', data=data.loc[train_mask & (data.Target.isin([0, 1])), :], 
           fit_reg=False);

sns.lmplot(x='AngleOfSign', y='SignHeight', hue='Target', data=data.loc[train_mask & (data.Target.isin([0, 1])), :], 
           fit_reg=False);

sns.lmplot(x='AngleOfSign', y='SignWidth', hue='Target', data=data.loc[train_mask & (data.Target.isin([0, 1])), :], 
           fit_reg=False);

lbl.classes_

def cross_validate(X, y, model, ret_fold_preds=False,
                   save_folds=False, plot_cv_scores=False):
    """
    Stratified K-Fold with 10 splits and then save each fold
    and analyze the performance of the model on each fold
    """
    
    skf = StratifiedKFold(n_splits=10, random_state=SEED)
    fold_counter = 0
    
    cv_scores = []
    preds     = []
    
    for (itr, ite) in tqdm_notebook(skf.split(X, y)):
        Xtr = X.iloc[itr]
        ytr = y.iloc[itr]
        
        Xte = X.iloc[ite]
        yte = y.iloc[ite]
        
        print('Class Distribution in the training fold \n', ytr.value_counts(normalize=True))
        print('Class Distribution in the test fold \n', yte.value_counts(normalize=True))
        
        
        if save_folds:
            save_file(pd.concat((Xtr, ytr), axis='columns'), '../data/processed/train_fold%s.feather'%(fold_counter))
            save_file(pd.concat((Xte, yte), axis='columns'), '../data/processed/test_fold%s.feather'%(fold_counter))
        
        print('Training model')
        start_time = time.time()
        model.fit(Xtr, ytr)
        end_time   = time.time()
        
        print('Took: {} seconds to train model'.format(end_time - start_time))
        
        start_time  = time.time()
        fold_preds  = model.predict_proba(Xte)
        
        if ret_fold_preds:
            preds.append(fold_preds)
        end_time    = time.time()
        
        print('Took: {} seconds to generate predictions'.format(end_time - start_time))
        
        fold_score = log_loss(yte, fold_preds)
        print('Fold log loss score: {}'.format(fold_score))
        
        cv_scores.append(fold_score)
        print('='*75)
        print('\n')
        
    if plot_cv_scores:
        plt.scatter(np.arange(0, len(cv_scores)), cv_scores)
    
    print('Mean cv score: {} \n Std cv score: {}'.format(np.mean(cv_scores), np.std(cv_scores)))
    
    return preds

def cv_multiple_models(X, y, feature_sets, models):
    skf = StratifiedKFold(n_splits=10, random_state=SEED)
    
    model_scores = [[] for _ in models]
    cv_scores    = []
    fold_index   = 0
    
    for (itr, ite) in tqdm_notebook(skf.split(X, y)):
        Xtr = X.iloc[itr]
        ytr = y.iloc[itr]
        
        Xte = X.iloc[ite]
        yte = y.iloc[ite]
        
        predictions = []
        
        for i, model in enumerate(models):
            print('Training model: {}'.format(i))
            curr_model = model.fit(Xtr.loc[:, feature_sets[i]], ytr)
            model_pred = curr_model.predict_proba(Xte.loc[:, feature_sets[i]])
            predictions.append(model_pred)
            model_scores[i].append(log_loss(yte, model_pred))
        
        predictions = np.array(predictions)
        vot = predictions[1]
        
        for i in range(1, len(predictions)):
            vot = vot + predictions[i]
        
        vot /= len(predictions)
        
        print('final ensemble predictions shape ', vot.shape)
        
        curr_metric = log_loss(yte, vot)
        cv_scores.append(curr_metric)
        print('split # {}, score = {}, models scores std = {}'            .format(fold_index, curr_metric,
            np.std([scr[fold_index] for scr in model_scores])))
        
        fold_index += 1
        
    print()
    print(cv_scores)
    print(np.mean(cv_scores), np.std(cv_scores))
    print()

diff_from_normal = 1 - data.SignAspectRatio
data = data.assign(diff_from_normal=diff_from_normal)

X = data.loc[train_mask, ['AngleOfSign', 'diff_from_normal', 'DetectedCamera', 'SignArea']]
y = data.loc[train_mask, 'Target']

Xtest = data.loc[~train_mask, ['AngleOfSign', 'diff_from_normal', 'DetectedCamera', 'SignArea']]

# train test splt
params = {
    'stratify': y,
    'test_size': .2,
    'random_state': SEED
}

X_train, X_test, y_train, y_test = train_test_split(X, y, **params)

model = RandomForestClassifier(n_estimators=500, max_depth=3, random_state=SEED)
params = {
    'ret_fold_preds': True,
    'save_folds': False,
    'plot_cv_scores': False
}

fold_preds_rf_single = cross_validate(X_train[['AngleOfSign']], y_train, model, **params)

model = RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_split=5, oob_score=True, random_state=SEED)

params = {
    'ret_fold_preds': True,
    'save_folds': False,
    'plot_cv_scores': False
}

fold_preds_rf = cross_validate(X_train[['DetectedCamera']], y_train, model, **params)

def calculate_correlation(fold_preds_rf, fold_preds_et):
    for i in tqdm_notebook(range(10)):
        print(pd.DataFrame(np.array(fold_preds_rf)[i]).corrwith(pd.DataFrame(np.array(fold_preds_et)[i])))
        print('='*75)
        print('\n')

calculate_correlation(fold_preds_rf, fold_preds_rf_single)

feature_sets = [['AngleOfSign'], ['AngleOfSign', 'diff_from_normal'],
                ['AngleOfSign', 'diff_from_normal'], ['DetectedCamera', 'diff_from_normal']
               ]

models = [RandomForestClassifier(n_estimators=500, n_jobs=-1, max_depth=3, random_state=SEED),
          ExtraTreesClassifier(n_estimators=750, n_jobs=-1, max_depth=7, random_state=SEED),
          xgb.XGBClassifier(seed=SEED), xgb.XGBClassifier(seed=SEED)
         ]

cv_multiple_models(X_train, y_train, feature_sets, models)



