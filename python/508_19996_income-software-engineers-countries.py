get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import os

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import geonamescache

from lxml import html

mpl.style.use('ramiro')

data_dir = os.path.expanduser('~/data')
gc = geonamescache.GeonamesCache()

chartinfo = '''Figures represent income data from May 2013 to May 2014 using exchange rates from May 2014. Average annual income figures are for 2014.
Author: Ramiro Gómez - ramiro.org • Data: Bloomberg/PayScale - bloomberg.com/visual-data/best-and-worst/highest-paid-software-engineers-countries'''

url ='http://www.bloomberg.com/visual-data/best-and-worst/highest-paid-software-engineers-countries'
xpath = '//table[@class="hid"]'
 
tree = html.parse(url)
table = tree.xpath(xpath)[0]
raw_html = html.tostring(table)

df = pd.read_html(raw_html, header=0, index_col=0)[0]
df.head()

df.dtypes

df['Country'] = df['Country'].apply(lambda x: x.rstrip('t').strip())

for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col].apply(lambda x: x.lstrip('$').replace(',', '')))

df.dtypes

ratio = round(df['Median annual pay for software engineers'] / df['Average annual income'], 2)
all(ratio == df['Ratio of median software engineer pay to average income'])

x = 'Median annual pay for software engineers'
y = 'Average annual income'
title = 'Median annual income of software engineers vs. general average in 50 countries'

fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111)
ax.plot(x, y, '.', data=df, ms=10, alpha=.7)
ax.set_title(title, fontsize=20, y=1.04)
ax.set_xlabel(x)
ax.set_ylim(bottom=0, top=101000)
ax.set_ylabel(y)

# Polynomial curve fitting 
# http://docs.scipy.org/doc/numpy/reference/routines.polynomials.classes.html
polynomial = np.polynomial.Polynomial.fit(df[x], df[y], 2)
xp = np.linspace(0, 120000, 100)
yp = polynomial(xp)
ax.plot(xp, yp, '-', lw=1, alpha=.5)

fig.text(0, -.04 , chartinfo, fontsize=11)
plt.show()

col = 'Ratio of median software engineer pay to average income'
title = 'Best and worst countries ranked by ratio of median software engineer pay to average income'
limit = 10

best = df.head(limit)[::-1]
worst = df.tail(limit)
ticks = np.arange(limit)

fig = plt.figure(figsize=(14, 5))
fig.suptitle(title, fontsize=20)

ax1 = fig.add_subplot(1, 2, 1)
ax1.barh(ticks, best[col], alpha=.5, color='#00ff00')
ax1.set_yticks(ticks)
ax1.set_yticklabels(best['Country'].values, fontsize=15, va='bottom')

ax2 = fig.add_subplot(1, 2, 2)
ax2.barh(ticks, worst[col], alpha=.5, color='#ff0000')
ax2.set_yticks(ticks)
ax2.set_yticklabels(worst['Country'].values, fontsize=15, va='bottom')

fig.text(0, -.07, chartinfo, fontsize=12)
plt.show()

df_map = df.copy()
names = gc.get_countries_by_names()
df_map['iso3'] = df_map['Country'].apply(lambda x: names[x]['iso3'])
df.head(5)

del df_map['Country']
df_map.to_csv(data_dir + '/economy/income-software-engineers-countries.csv', encoding='utf-8', index=False)

signature

