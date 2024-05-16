get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import itertools
import math

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('ramiro')

df = pd.read_excel('csv/top-incomes.xlsx', 1, skiprows=1)

chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: World Top Incomes Database - parisschoolofeconomics.eu'
infosize = 13

df.head()

income_col = 'Top 1% income share'
cols = ['Year', 'Country', income_col]
df_top = df[cols].dropna(axis=0, how='any')

year_counts = df_top.Year.value_counts()
sufficient = year_counts[year_counts > 15]
sufficient.sort_index(ascending=False).head()

year = 2010
title = 'Income share of the top 1% earners across countries in {}'.format(year)

df_top_year = df_top[df_top['Year'] == year].sort(columns=[income_col])

s = df_top_year.set_index('Country')[income_col]
ax = s.plot(
    kind='barh',
    figsize=(12, 8), 
    title=title)
ax.tick_params(labelbottom='off')
ax.set_ylabel('', visible=False)

for i, label in enumerate(s):
    ax.annotate(str(label) + '%', (label + .2, i - .15))

plt.annotate(chartinfo, xy=(0, -1.04), xycoords='axes fraction', fontsize=infosize)
plt.savefig('img/income-share-top1-{}.png'.format(year), bbox_inches='tight')

df_pivot = df_top.pivot(*cols)

num_countries = len(df_pivot.columns)
xmax = max(df_pivot.index)
xmin = xmax - 100

ncols = 5
nrows = math.ceil(num_countries / ncols)

title = 'Income share of the top 1% earners in {:d} countries between {:d} and {:d}'.format(num_countries, xmin, xmax)
footer = 'Included are countries with at least one record for the top 1% income share in the given time range.\n' + chartinfo

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
fig.suptitle(title, y=1.03, fontsize=20)
fig.set_figwidth(12)
fig.set_figheight(11)

for idx, coords in enumerate(itertools.product(range(nrows), range(ncols))):
    ax = axes[coords[0], coords[1]]
    country = df_pivot.columns[idx]
    df_pivot[country].plot(
        ax=ax,
        xlim=(xmin, xmax)
    )
    
    ax.set_title(country, fontsize=15)
    ax.set_xlabel('', visible=False)
    ax.set_xticks(list(range(xmin, xmax + 1, 25)))
    ax.tick_params(pad=10, labelsize=11)
    ax.tick_params(labelsize=11)
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda val, p: '{}%'.format(int(val))))


fig.text(0, -.03, footer, fontsize=infosize)
fig.tight_layout()

plt.savefig('img/income-share-top1-{}-countries-{}-{}.png'.format(num_countries, xmin, xmax), bbox_inches='tight')

cols = [
    'Top 0.1% average income',
    'Top 0.1% average income-including capital gains', 
    'Top 1% average income',
    'Top 1% average income-including capital gains'
]
highlight = 'Top 0.1% average income-including capital gains'

df_us = df[df['Country'] == 'United States']
df_us.set_index('Year', inplace=True)

xmax = df_us.index.max()
xmin = xmax - 100

title = 'Evolution of the top incomes in the US from {} to {}'.format(xmin, xmax)

ax = df_us[cols].plot(figsize=(12, 10), title=title)

ax.yaxis.set_major_formatter(
    mpl.ticker.FuncFormatter(lambda val, p: format(int(val), ',')))

ax.set_ylabel('Real 2014 US Dollars', fontsize=13)
ax.set_xlabel('', visible=False)
ax.legend(loc=2, prop={'size': 12})
plt.annotate(chartinfo, xy=(0, -1.1), xycoords='axes fraction', fontsize=infosize)

plt.annotate(
    'Dot-com bust', 
    xy=(2000, df_us.ix[2000][highlight]),
    xytext=(0, 5),
    textcoords='offset points',
    ha='center',
    va='bottom',
    size=11)

plt.annotate(
    'Financial crisis', 
    xy=(2007, df_us.ix[2007][highlight]),
    xytext=(0, 5),
    textcoords='offset points',
    ha='center',
    va='bottom',
    size=11)

plt.savefig('img/income-share-top-us-{}-{}.png'.format(xmin, xmax), bbox_inches='tight')

get_ipython().magic('signature')

