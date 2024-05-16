get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext signature')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import helpers

# Set style and meta info.
mpl.style.use('ramiro')
mpl.rcParams['axes.grid'] = False

# DAYOFWEEK returns the day of the week as an integer between 1 (Sunday) and 7 (Saturday).
weekdays_short = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: /u/Stuck_In_the_Matrix & /u/fhoffa - reddit.com'
infosize = 12

df = pd.read_csv('data/reddit/reddit-top-posts-by-subreddit-weekday-hour.csv')
df.head()

def plot_post_times(subreddit=None):
    df_plot = df.copy()
    title = 'Number of reddit submissions that reached ≥1000 points by time of submission'
    footer = 'Accounts for {:,d} submissions to subreddits with >100 submissions that reached at least 1000 points from January 2006 to August 31, 2015.\n'.format(df.num_with_min_score.sum())
    
    if subreddit:
        df_plot = df[df.subreddit.str.lower() == subreddit.lower()]
        title = 'Number of submissions to /r/{} that reached ≥ 1000 points by time of submission'.format(subreddit)
        footer = 'Accounts for {:,d} submissions to /r/{} that reached at least 1000 points from the subreddit\'s start to August 31, 2015.\n'.format(int(df_plot[0:1].total), subreddit)
    
    grouped = df_plot.groupby(['dayofweek', 'hourofday'])['num_with_min_score'].sum()
    if grouped.empty:
        print('Empty series after grouping.')
        return
    
    image = grouped.unstack()

    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = plt.cm.Greens
    img = ax.imshow(image, cmap=cmap, interpolation='nearest')
    
    # Annotations, labels, and axes ticks.
    ax.set_title(title, y=1.08, fontsize=16)
    ax.annotate(footer + chartinfo, xy=(0, -.35), xycoords='axes fraction', fontsize=infosize)
    ax.set_xlabel('Hour of reddit submission (UTC)', fontsize=infosize)
    ax.set_ylabel('Weekday of reddit submission', fontsize=infosize)
    plt.xticks(range(24))
    plt.yticks(range(7), weekdays_short)

    # Draw color legend.
    values = grouped.values
    bins = np.linspace(values.min(), values.max(), 5)
    plt.colorbar(img, shrink=.6, ticks=bins)

    plt.savefig('img/' + helpers.slug(title), bbox_inches='tight')

plot_post_times()

for sub in ['DataIsBeautiful', 'linux', 'MapPorn', 'photoshopbattles', 'ProgrammerHumor', 'programming', 'soccer', 'TodayILearned']:
    plot_post_times(sub)

get_ipython().magic('signature')

