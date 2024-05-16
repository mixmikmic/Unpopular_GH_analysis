import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

df_vio = pd.read_csv('data/training_labels.txt', parse_dates=['date'])
df_vio.rename(columns={'date':'inspect_date'}, inplace=True)
df_vio.head()

from helper_methods import drop_duplicate_inspections
df_vio = df_vio.sort_values(['restaurant_id', 'inspect_date'])
df_vio = drop_duplicate_inspections(df_vio, threshold=60)
df_vio.head()

def mean_violations_up_to_inspection_date(x, r, d):
    xf = x[(x['restaurant_id'] == r) & (x['inspect_date'] < d)]
    return xf.mean()['*'] if not xf.empty else 3.0

df_vio['mean_one'] = df_vio.apply(lambda row: mean_violations_up_to_inspection_date(df_vio, row['restaurant_id'], row['inspect_date']), axis=1)
mean_one_max = df_vio['mean_one'].max()
df_vio['mean_one_sc'] = df_vio['mean_one'].apply(lambda x: x / mean_one_max)

trans = pd.read_csv('data/restaurant_ids_to_yelp_ids.csv')
trans = trans[trans['yelp_id_1'].isnull()]
trans.drop(['yelp_id_1', 'yelp_id_2', 'yelp_id_3'], axis=1, inplace=True)
trans.columns = ['restaurant_id', 'business_id']
trans.head()

df_trans = pd.merge(trans, df_vio, on='restaurant_id', how='inner')
df_trans.head()

from helper_methods import read_json

df_biz = read_json('data/yelp_academic_dataset_business.json')
df_biz.head(2).transpose()

neighborhoods = list(set(df_biz['neighborhoods'].sum()))
for neighborhood in neighborhoods:
    df_biz[neighborhood] = df_biz.neighborhoods.apply(lambda x: 1.0 if neighborhood in x else 0.0)
df_biz['Other'] = df_biz.neighborhoods.apply(lambda x: 1.0 if x == [] else 0.0)
neighborhoods += ['Other']
df_biz[['neighborhoods'] + neighborhoods].head(5).transpose()

# it was necessary to add Other so that each restaurant was assigned
df_biz[neighborhoods].sum(axis=0).sort_index()

# every restaurant is assigned to at least 1 neighborhood which may be Other
df_biz[neighborhoods].sum(axis=1).value_counts()

categories = list(set(df_biz['categories'].sum()))
for category in categories:
    df_biz[category] = df_biz.categories.apply(lambda x: 1.0 if category in x else 0.0)
df_biz[['categories'] + categories].head(3).transpose()

df_biz[categories].sum(axis=1).value_counts().sort_index()

avg_violations = df_trans.groupby('business_id').agg({'*':[np.size, np.mean, np.std]})
avg_violations.columns = ['size', 'mean-*', 'std-*']
std_mean = avg_violations.mean()[2]
avg_violations['std-*'] = avg_violations['std-*'].apply(lambda x: std_mean if np.isnan(x) else x)

df_biz = pd.merge(avg_violations, df_biz, left_index=True, right_on='business_id', how='inner')
df_biz['mean-*'] = df_biz['mean-*'].apply(lambda x: x / df_biz['mean-*'].max())
df_biz['std-*'] = df_biz['std-*'].apply(lambda x: x / df_biz['std-*'].max())

cd = pd.read_csv('crime_density.csv', names=['crime_density', 'business_id', 'stars'])
df_biz = pd.merge(cd, df_biz, on='business_id', how='inner')

df_cmb = pd.merge(df_trans, df_biz, on='business_id', how='inner')
df_cmb.head(2).transpose()

np.random.seed(0)
msk = np.random.rand(df_cmb.shape[0]) < 0.8
df_train = df_cmb[msk]
df_test = df_cmb[~msk]

from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse

def get_scores(train, test, columns):
    X_train = train[columns].values
    y_true_train = train['*'].values

    param_grid = {'alpha':np.logspace(-4, 2, num=25, base=10)}
    gs = GridSearchCV(Lasso(), param_grid, scoring='mean_squared_error', cv=10)
    gs = gs.fit(X_train, y_true_train)

    y_pred_train = gs.predict(X_train)
    y_true_test = test['*'].values
    y_pred_test = gs.predict(df_test[columns].values)

    mse_train = mse(y_true_train, y_pred_train)
    mse_test = mse(y_true_test, y_pred_test)
    
    return 'MSE (train) = %.1f, MSE (test) = %.1f' % (mse_train, mse_test)

print get_scores(df_train, df_test, ['mean_one_sc'])

print get_scores(df_train, df_test, ['Chinese'])

print get_scores(df_train, df_test, ['Back Bay'])

print get_scores(df_train, df_test, neighborhoods)

print get_scores(df_train, df_test, categories)

print get_scores(df_train, df_test, neighborhoods + categories)

print get_scores(df_train, df_test, ['mean_one_sc'] + categories)

print get_scores(df_train, df_test, ['mean_one_sc'] + categories + neighborhoods)

