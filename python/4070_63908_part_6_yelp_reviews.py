import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

df = pd.read_csv('data/training_labels.txt', parse_dates=['date'])
df.rename(columns={'date':'inspect_date'}, inplace=True)
df['weighted_violations'] = 1 * df['*'] + 3 * df['**'] + 5 * df['***']
df.head()

df.info()

from helper_methods import drop_duplicate_inspections
df = df.sort_values(['restaurant_id', 'inspect_date'])
df = drop_duplicate_inspections(df, threshold=60)
df.head()

df.info()

for column in df:
    print df.shape[0], df[column].unique().size, column

trans = pd.read_csv('data/restaurant_ids_to_yelp_ids.csv')
trans = trans[trans['yelp_id_1'].isnull()]
trans.drop(['yelp_id_1', 'yelp_id_2', 'yelp_id_3'], axis=1, inplace=True)
trans.columns = ['restaurant_id', 'business_id']
trans.head()

trans.info()

# uncomment the lines below to work with restaurants that have had multiple owners
# the problem with this is that it is not possible to determine when ownership
# changed

#from helper_methods import biz2yelp
#trans = biz2yelp()
#trans.columns = ['restaurant_id', 'business_id']
#trans.head()

df_trans = pd.merge(trans, df, on='restaurant_id', how='inner')
df_trans.head()

df_trans.info()

np.random.seed(0)
msk = np.random.rand(train_test.shape[0]) < 0.8
df_train = df_trans[msk]
df_test = df_trans[~msk]

from sklearn.metrics import mean_squared_error as mse

y_true_train = df_train['*'].values
y_pred_train = 3 * np.ones(df_train.shape[0])

y_true_test = df_test['*'].values
y_pred_test = 3 * np.ones(df_test.shape[0])

print mse(y_true_train, y_pred_train)
print mse(y_true_test, y_pred_test)

avg_violations = df_trans.groupby('restaurant_id').agg({'*': [np.size, np.mean, np.sum], '**': [np.mean, np.sum], '***': [np.mean, np.sum], 'weighted_violations': [np.mean, np.sum]})
avg_violations.head(5)

avg_vio_train = pd.merge(avg_violations, df_train, left_index=True, right_on='restaurant_id', how='right')
y_pred_train = avg_vio_train[[('*', 'mean')]].values
y_true_train = avg_vio_train[['*']].values

avg_vio_test = pd.merge(avg_violations, df_test, left_index=True, right_on='restaurant_id', how='right')
y_pred_test = avg_vio_test[[('*', 'mean')]].values
y_true_test = avg_vio_test[['*']].values

print mse(y_true_train, y_pred_train)
print mse(y_true_test, y_pred_test)

import re
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

stops = stopwords.words("english")
def review_to_words_porter(raw_review):
    review_text = BeautifulSoup(raw_review, 'lxml').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    porter = PorterStemmer()
    return " ".join(porter.stem(word) for word in words if word not in stops)

from helper_methods import read_json
df_rev = read_json('data/yelp_academic_dataset_review.json')
df_rev.rename(columns={'date':'review_date'}, inplace=True)
df_rev['text'] = df_rev['text'].apply(lambda x: review_to_words_porter(x))
df_rev.head(3)

avg_stars = df_rev.groupby('business_id').agg({'stars':[np.mean, np.sum]})
avg_stars.head()

stars_violations = pd.merge(avg_vio_id, avg_stars, left_on='business_id', right_index=True, how='inner')
stars_violations.head(3)

plt.plot(stars_violations[('weighted_violations', 'mean')], stars_violations[('stars', 'mean')], '.')
plt.xlabel('Number of mean weighted violations')
plt.ylabel('Mean star rating')
plt.xlim(0, 30)

from scipy.stats import pearsonr
pearsonr(stars_violations[('weighted_violations', 'mean')], stars_violations[('stars', 'mean')])

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plt.plot(avg_stars[('stars', 'sum')], avg_stars[('stars', 'mean')], '.')
plt.xlabel('Number of reviews')
plt.ylabel('Average star rating')
plt.xlim(0, 5000)

pearsonr(avg_stars[('stars', 'sum')], avg_stars[('stars', 'mean')])

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plt.hist(avg_stars[('stars', 'sum')], bins=250, range=(0, 250))
plt.xlabel('Number of reviews')
plt.ylabel('Number of restaurants')

# number of days in the review window 
t_days = 14

df_id = pd.merge(df, trans, on='restaurant_id', how='inner')
df_id.head()

df_rev[['business_id', 'text', 'review_date']].head()

xl = pd.merge(df_id, df_rev, on='business_id', how='outer')
xl = xl[((xl['inspect_date'] - xl['review_date']) / np.timedelta64(1, 'D') > 0) & ((xl['inspect_date'] - xl['review_date']) / np.timedelta64(1, 'D') <= t_days)]
xl.drop(['id', 'weighted_violations', 'business_id', 'review_id', 'votes', 'type', 'user_id'], axis=1, inplace=True)
xl.head()

xl.info()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 1), smooth_idf=True, norm='l2')
X_train = tfidf.fit_transform(xl['text'].values)
y_train = xl['*'].values

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse

linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_train_pred = linreg.predict(X_train)
print 'Linear model: mse =', mse(y_train, y_train_pred)

#rf = RandomForestRegressor(n_estimators=10, criterion='mse')
#rf.fit(X_train, y_train)
#y_train_pred = rf.predict(X_train)
#print 'RF model: mse =', mse(y_train, y_train_pred)

#for x in sorted(zip(linreg.coef_, count.vocabulary_)):
#    print x

t_days = 60

cutoff = df_rev['review_date'].sort_values(ascending=False)[:5]
cutoff[:5]

cutoff = cutoff.iloc[0]
cutoff

df_sub = pd.read_csv('PhaseIISubmissionFormat.csv')
df_sub['date'] = pd.to_datetime(df_sub['date'])
df_sub.rename(columns={'date':'inspect_date'}, inplace=True)
df_sub.head()

df_sub.info()

for column in df_sub.columns:
    print df_sub.shape[0], df_sub[column].unique().size, column

df_sub = pd.merge(df_sub, trans, on='restaurant_id', how='left')
df_sub.head()

df_sub.info()

for column in df_sub.columns:
    print df_sub.shape[0], df_sub[column].unique().size, column

bg = pd.merge(df_sub, df_rev, on='business_id', how='left')
bg = bg[((cutoff - bg['review_date']) / np.timedelta64(1, 'D') < t_days)]
bg.drop(['id', 'business_id', 'review_id', 'stars', 'votes', 'type', 'user_id'], axis=1, inplace=True)
bg.head(3)

bg.info()

for column in bg.columns:
    print bg.shape[0], bg[column].unique().size, column

bg_by_restaurant = bg.groupby('restaurant_id').size()
print 'min:', bg_by_restaurant.min(), '  max:', bg_by_restaurant.max()

bg['*-predict'] = linreg.predict(tfidf.transform(bg['text'].values))
bg.head()

mean_violations = bg.groupby('restaurant_id').agg({'*-predict':[np.mean]})
pred = pd.merge(df_sub, mean_violations, left_on='restaurant_id', right_index=True, how='left')
pred.head()

pred.info()

pred[('*-predict', 'mean')].fillna(3, inplace=True)
pred[('*-predict', 'mean')] = pred[('*-predict', 'mean')].apply(lambda x: int(round(x)))
pred.head()

