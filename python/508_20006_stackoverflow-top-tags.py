get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import itertools
import math

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ramiro')

df = pd.concat([
    pd.read_csv('data/stackoverflow/Aggregated post stats for top tags per day and tag 0 - 10.csv'),
    pd.read_csv('data/stackoverflow/Aggregated post stats for top tags per day and tag 10 - 20.csv')
])

chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: StackExchange - data.stackexchange.com/stackoverflow'
infosize = 13

df.describe()

col_labels = ['Posts', 'Views', 'Answers', 'Comments', 'Favorites', 'Score']

df.dtypes

df.post_date = pd.to_datetime(df.post_date)

df[df.tag_name == 'java'].set_index('post_date').posts_per_day.plot()

grouped_by_tag = df.groupby('tag_name').agg('sum')

grouped_by_tag.columns = ['Total {}'.format(l) for l in col_labels]

grouped_by_tag.plot(subplots=True, figsize=(12, 10), kind='bar', legend=False)
plt.show()

grouped_by_date = df.groupby('post_date').agg('sum')
grouped_by_date.plot(subplots=True, figsize=(12, 12))

from pandas.tools.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(12, 12), diagonal='kde')

grouped_by_date.posts_per_day.plot()

grouped_by_date.answers_per_day.plot()

grouped_by_date.favorites_per_day.plot()

grouped_by_date.score_per_day.plot()

grouped_by_date.sort('score_per_day', ascending=False)

df.tag_name.value_counts()



