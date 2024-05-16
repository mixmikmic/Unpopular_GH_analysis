get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ramiro')

chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: Movie Body Counts - moviebodycounts.com'

df = pd.read_csv('http://files.figshare.com/1332945/film_death_counts.csv')

df.head()

df.columns = ['Film', 'Year', 'Body count', 'MPAA', 'Genre', 'Director', 'Minutes', u'IMDB']

df['Film count'] = 1
df['Body count/min'] = df['Body count'] / df['Minutes'].astype(float)
df.head()

group_year = df.groupby('Year').agg([np.mean, np.median, sum])
group_year.tail()

df_bc = pd.DataFrame({'mean': group_year['Body count']['mean'],
                      'median': group_year['Body count']['median']})

df_bc_min = pd.DataFrame({'mean': group_year['Body count/min']['mean'], 
                          'median': group_year['Body count/min']['median']})

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(16, 22))

group_year['Film count']['sum'].plot(kind='bar', ax=axes[0]); axes[0].set_title('Film Count')
group_year['Body count']['sum'].plot(kind='bar', ax=axes[1]); axes[1].set_title('Total Body Count')
df_bc.plot(kind='bar', ax=axes[2]); axes[2].set_title('Body Count by Film')
df_bc_min.plot(kind='bar', ax=axes[3]); axes[3].set_title('Body Count by Minute')

for i in range(4):
    axes[i].set_xlabel('', visible=False)
    
plt.annotate(chartinfo, xy=(0, -1.2), xycoords='axes fraction')

df_film = df.set_index('Film')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))

bc = df_film.sort('Body count')['Body count'].tail(10)
bc.plot(kind='barh', ax=axes[0])
axes[0].set_title('Total Body Count')

bc_min = df_film.sort('Body count/min')['Body count/min'].tail(10)
bc_min.plot(kind='barh', ax=axes[1])
axes[1].set_title('Body Count per Minute')
axes[1].yaxis.set_ticks_position('right')

for i in range(2):
    axes[i].set_ylabel('', visible=False)
    
plt.annotate(chartinfo, xy=(0, -1.07), xycoords='axes fraction')

from IPython.display import IFrame
IFrame('https://www.youtube-nocookie.com/embed/HdNn5TZu6R8', width=800, height=450)

df[df['Director'].apply(lambda x: -1 != x.find('|'))].head()

def expand_col(df_src, col, sep='|'):
    di = {}
    idx = 0
    for i in df_src.iterrows():
        d = i[1]
        names = d[col].split(sep)
        for name in names:
            # operate on a copy to not overwrite previous director names
            c = d.copy()
            c[col] = name
            di[idx] = c
            idx += 1

    df_new = pd.DataFrame(di).transpose()
    # these two columns are not recognized as numeric
    df_new['Body count'] = df_new['Body count'].astype(float)
    df_new['Body count/min'] = df_new['Body count/min'].astype(float)
    
    return df_new

df_dir = expand_col(df, 'Director')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))

bc_sum = df_dir.groupby('Director').sum().sort('Body count').tail(10)
bc_sum['Body count'].plot(kind='barh', ax=axes[0])
axes[0].set_title('Total Body Count')

bc_mean = df_dir.groupby('Director').agg(np.mean).sort('Body count/min').tail(10)
bc_mean['Body count/min'].plot(kind='barh', ax=axes[1])
axes[1].set_title('Body Count per Minute')
axes[1].yaxis.set_ticks_position('right')

for i in range(2):
    axes[i].set_ylabel('', visible=False)

plt.annotate(chartinfo, xy=(0, -1.07), xycoords='axes fraction')

df_genre = expand_col(df, 'Genre')
df_genre['Genre'].value_counts().plot(kind='bar', figsize=(12, 6), title='Genres by film count')

plt.annotate(chartinfo, xy=(0, -1.28), xycoords='axes fraction')

bc_mean = df_genre.groupby('Genre').agg(np.mean).sort('Body count/min', ascending=False)
ax = bc_mean['Body count/min'].plot(kind='bar', figsize=(12, 6), title='Genres by body count per minute')
ax.set_xlabel('', visible=False)
plt.annotate(chartinfo, xy=(0, -1.32), xycoords='axes fraction')

df_genre[(df_genre['Genre'] == 'War') | (df_genre['Genre'] == 'History')].sort('Body count/min', ascending=False).head(20)

ratings = df['MPAA'].value_counts()
ratings

rating_names = ratings.index
rating_index = range(len(rating_names))
rating_map = dict(zip(rating_names, rating_index))
mpaa = df['MPAA'].apply(lambda x: rating_map[x])

fig, ax = plt.subplots(figsize=(14, 10))
ax.scatter(mpaa, df['Body count/min'], s=df['Body count'], alpha=.5)
ax.set_title('Body counts and MPAA ratings')
ax.set_xlabel('MPAA Rating')
ax.set_xticks(rating_index)
ax.set_xticklabels(rating_names)
ax.set_ylabel('Body count per minute')
plt.annotate(chartinfo, xy=(0, -1.12), xycoords='axes fraction')

bc_top = df.sort('Body count', ascending=False)[:3]
annotations = []
for r in bc_top.iterrows():
    annotations.append([r[1]['Film'], r[1]['IMDB'], r[1]['Body count/min']])

fig, ax = plt.subplots(figsize=(14, 10))
ax.scatter(df['IMDB'], df['Body count/min'], s=df['Body count'], alpha=.5)
ax.set_title('Body count and IMDB ratings')
ax.set_xlabel('IMDB Rating')
ax.set_ylabel('Body count per minute')

for annotation, x, y in annotations:
    plt.annotate(
        annotation,
        xy=(x, y),
        xytext=(0, 30),
        textcoords='offset points',
        ha='center',
        va='bottom',
        size=12.5,
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='-'))

plt.annotate(chartinfo, xy=(0, -1.12), xycoords='axes fraction')

get_ipython().magic('signature')

