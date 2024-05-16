get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import helpers

plt.style.use('ramiro')

df = pd.read_csv('csv/bundesliga.csv', parse_dates=['date'], encoding='latin-1')

chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: James Curley - github.com/jalapic/engsoccerdata'
infosize = 13

df.head()

df.describe()

def result(row):
    if row.hgoal > row.vgoal:
        return 'Home win'
    elif row.hgoal < row.vgoal:
        return 'Home loss'
    return 'Draw'

df['resulttype'] = df.apply(result, axis=1)
resulttypes_by_season = df.groupby(['Season', 'resulttype']).agg(['count'])['date']
df_rs = resulttypes_by_season.unstack()
df_rs.head()

df_rs = df_rs.apply(lambda x: 100 * x / float(x.sum()), axis=1)
df_rs.head()

def season_display(year):
    s = str(year)
    return '{0}/{1:02d}'.format(year, int(str(year + 1)[-2:]))

colors = ['#993300', '#003366', '#99cc99']
alpha = .7

c1, c2, c3 = df_rs['count'].columns
s1 = df_rs['count'][c1]
s2 = df_rs['count'][c2]
s3 = df_rs['count'][c3]

xmax = df_rs.index.max()
xmin = df_rs.index.min()

title = 'Wins, losses and draws in the Bundesliga seasons {} to {} in percent'.format(
    season_display(xmin), season_display(xmax))

ax = df_rs.plot(kind='bar', stacked=True, figsize=(16, 6), color=colors, title=title, fontsize=13, width=1, alpha=alpha)
ax.set_ylim([0, 100])
ax.set_xlabel('', visible=False)
ax.xaxis.set_major_formatter(
    mpl.ticker.FuncFormatter(lambda val, p: season_display(df_rs.index[val])))

p1 = mpatches.Patch(color=colors[0], label=c1, alpha=alpha)
p2 = mpatches.Patch(color=colors[1], label=c2, alpha=alpha)
p3 = mpatches.Patch(color=colors[2], label=c3, alpha=alpha)

ax.legend(loc=(.69, -.23), handles=[p1, p2, p3], ncol=3, fontsize=13)
ax.annotate(chartinfo, xy=(0, -1.21), xycoords='axes fraction', fontsize=infosize)
plt.savefig('img/{}.png'.format(helpers.slug(title)), bbox_inches='tight')

df_2points = df[df.Season < 1995]
df_3points = df[df.Season >= 1995]

results_2points = df_2points.FT.value_counts()
results_3points = df_3points.FT.value_counts()

df_results = pd.concat([
    results_2points / results_2points.sum(), 
    results_3points / results_3points.sum()], axis=1).fillna(0)

limit = 30
title = '{} most common Bundesliga results: 2 vs 3 points for a win'.format(limit)
ylabel = 'Relative frequency of result'

cols = ['2 points for a win', '3 points for a win']
df_results.columns = cols

df_results['sum'] = df_results[cols[0]] + df_results[cols[1]]
df_results.sort('sum', inplace=True, ascending=False)

ax = df_results[cols].head(limit).plot(kind='bar', figsize=(16, 6), title=title)
ax.set_xticklabels(df_results.index[:limit], rotation=0)
ax.set_ylabel(ylabel)
ax.yaxis.set_major_formatter(
      mpl.ticker.FuncFormatter(lambda val, p: '{}%'.format(int(val*100))))

text = '''The 30 most common full time results from {} to {} in the Germand Bundesliga, 2-points rule before 1995/96 and 3-points rule thereafter.
{}'''.format(season_display(xmin), season_display(xmax), chartinfo)
ax.annotate(text, xy=(0, -1.16), xycoords='axes fraction', fontsize=infosize)
plt.savefig('img/{}.png'.format(helpers.slug(title)), bbox_inches='tight')

signature

