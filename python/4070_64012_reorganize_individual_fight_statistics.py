import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)

iofile = 'data/fightmetric_individual_fights/detailed_stats_individual_fights_RAW.csv'
df = pd.read_csv(iofile, header=0, parse_dates=['Date'])
df.head(3)

df.info()

df[pd.isnull(df.Fighter1)].shape

# rename the second instance
idx = df[(df.Fighter1 == 'Dong Hyun Kim') & (df.Fighter2 == "Brendan O'Reilly")].index.values
df = df.set_value(idx, 'Fighter1', 'Dong Hyun Kim 2')
idx = df[(df.Fighter2 == 'Dong Hyun Kim') & (df.Fighter1 == 'Polo Reyes')].index.values
df = df.set_value(idx, 'Fighter2', 'Dong Hyun Kim 2')
idx = df[(df.Fighter2 == 'Dong Hyun Kim') & (df.Fighter1 == 'Dominique Steele')].index.values
df = df.set_value(idx, 'Fighter2', 'Dong Hyun Kim 2')

ftr = 'Dong Hyun Kim 2'
df[(df.Fighter1 == ftr) | (df.Fighter2 == ftr)]

df.describe()

df = df.dropna()
df.shape

iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)

fights.shape

not_analyzed = []
for index, row in fights.iterrows():
     fighter1 = row['Winner']
     fighter2 = row['Loser']
     msk1 = ((df.Fighter1 == fighter1) & (df.Fighter2 == fighter2))
     msk2 = ((df.Fighter1 == fighter2) & (df.Fighter2 == fighter1))
     x = df[msk1 & (df.Date == row['Date'])].shape[0]
     y = df[msk2 & (df.Date == row['Date'])].shape[0]
     if x + y == 0:
          not_analyzed.append(row.values)
pd.DataFrame(not_analyzed)

ftr = 'Kazushi Sakuraba'
fights[(fights.Winner == ftr) | (fights.Loser == ftr)]

xf = []
for index, row in df.iterrows():
     fighter1 = row['Fighter1']
     fighter2 = row['Fighter2']
     msk1 = ((fights.Winner == fighter1) & (fights.Loser == fighter2))
     msk2 = ((fights.Winner == fighter2) & (fights.Loser == fighter1))
     x = fights[msk1 & (fights.Date == row['Date'])].shape[0]
     y = fights[msk2 & (fights.Date == row['Date'])].shape[0]
     if (x == 1):
          xf.append(list(row.values))
     elif (y == 1):
          xf.append([row[0]] + list(row[12:23].values) + list(row[1:12].values))
     else:
          print 'Sakuraba fought Silveira twice in one night'
          xf.append(list(row.values))

xf = pd.DataFrame(xf, columns=df.columns)
xf.head(10)

xf.shape

# fx is very large since cartesian product so do filter after
fx = fights.merge(xf, left_on='Winner', right_on='Fighter1', how='left')
fx = fx[(fx.Date_x == fx.Date_y) & (fx.Loser == fx.Fighter2)]
fx.shape

ftr = 'Kazushi Sakuraba'
fx[(fx.Winner == ftr) | (fx.Loser == ftr)]

# rename the second instance
idx = fx[(fx.Winner == ftr) & (fx.Outcome == 'def.') & (fx.SigStrikesLanded1 == 0)].index.values
fx = fx.drop(idx, axis=0)
idx = fx[(fx.Winner == ftr) & (fx.Outcome == 'no contest') & (fx.SigStrikesLanded1 == 1)].index.values
fx = fx.drop(idx, axis=0)
fx.shape

ftr = 'Kazushi Sakuraba'
fx[(fx.Winner == ftr) | (fx.Loser == ftr)]

fx = fx.drop(['Fighter1', 'Fighter2', 'Date_y'], axis=1)
new_cols = []
for column in fx.columns:
      new_cols.append((column, column.replace('1', '').replace('2', '_L').replace('Date_x', 'Date')))
fx = fx.rename(columns=dict(new_cols))
fx.head(3)

fx.to_csv('data/fightmetric_individual_fights/detailed_stats_individual_fights_FINAL.csv', index=False)

fx[pd.isnull(fx.Knockdowns)]

