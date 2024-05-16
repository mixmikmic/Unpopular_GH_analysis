get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext signature')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import helpers

# Set style and meta info.
mpl.style.use('ramiro')

chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: Reddit /u/Stuck_In_the_Matrix & /u/fhoffa - reddit.com'
infosize = 12

df_subreddit_ranks = pd.read_csv('csv/reddit_comments_201505_subreddit_ranks.csv', index_col='subreddit')
df_gilded = pd.read_csv('csv/reddit_comments_201505_gilded_comments.csv')
df_gilded.head()

df_gilded.columns

df_gilded_subreddits = df_gilded.groupby('subreddit').agg('count')
df_gilded_ranks = df_gilded_subreddits.join(df_subreddit_ranks)

df_gilded_ranks['gilded_ratio'] = df_gilded_ranks.gilded / df_gilded_ranks.comments
df_gilded_ranks.sort('gilded_ratio', ascending=False).head(10)['gilded_ratio'].plot(kind='barh', figsize=(6, 4))

df_gilded.link_id.value_counts()

df_gilded.describe()

df_gilded.sort('gilded', ascending=False).head(10)

