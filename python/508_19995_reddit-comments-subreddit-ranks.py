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

df = pd.read_csv('csv/reddit_comments_201505_subreddit_ranks.csv', index_col='subreddit')
df.head()

limit = 30
title = 'Top {} Subreddits ranked by number of comments in May 2015'.format(limit)

s = df.sort('comments', ascending=False).head(30)['comments'].order()
ax = s.plot(kind='barh', figsize=(10, 12), title=title)

ax.tick_params(labelbottom='off')
ax.yaxis.set_label_text('')
ax.annotate(chartinfo, xy=(0, -1.02), xycoords='axes fraction', fontsize=infosize)

for i, value in enumerate(s):
    label = format(int(value), ',')
    ax.annotate(label, (value + 30000, i - .14))

plt.savefig('img/' + helpers.slug(title), bbox_inches='tight')

title = 'Top {} Subreddits ranked by number of authors in May 2015'.format(limit)

s = df.sort('authors', ascending=False).head(30)['authors'].order()
ax = s.plot(kind='barh', figsize=(10, 12), title=title)

ax.tick_params(labelbottom='off')
ax.yaxis.set_label_text('')
ax.annotate(chartinfo, xy=(0, -1.02), xycoords='axes fraction', fontsize=infosize)

for i, value in enumerate(s):
    label = format(int(value), ',')
    ax.annotate(label, (value + 5000, i - .14))

plt.savefig('img/' + helpers.slug(title), bbox_inches='tight')

title = 'Top {} Subreddits with the highest ratio of comments by author in May 2015'.format(limit)

df['comment_author_ratio'] = df['comments'] / df['authors']

s = df.sort('comment_author_ratio', ascending=False).head(limit)['comment_author_ratio'].order()
ax = s.plot(kind='barh', figsize=(10, 12), title=title)

ax.tick_params(labelbottom='off')
ax.yaxis.set_label_text('')
ax.annotate(chartinfo, xy=(0, -1.02), xycoords='axes fraction', fontsize=infosize)

for i, value in enumerate(s):
    label = format(value, ',.2f')
    ax.annotate(label, (value + 15, i - .15))

plt.savefig('img/' + helpers.slug(title), bbox_inches='tight')

get_ipython().magic('signature')

