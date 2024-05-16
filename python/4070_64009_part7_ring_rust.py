import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')
from scipy.stats import t

iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)

fights = fights[(fights.Date > pd.to_datetime('2005-01-01')) & (fights.Outcome != 'no contest')]

from collections import defaultdict

wins = defaultdict(int)
total = defaultdict(int)
wins2 = defaultdict(int)
total2 = defaultdict(int)
wins_12 = 0
total_12 = 0
win_lose = fights.Winner.append(fights.Loser).unique()
for fighter in win_lose:
     msk = (fights.Winner == fighter) | (fights.Loser == fighter)
     all_fights = fights[msk].sort_values('Date').reset_index()
     for i in range(0, all_fights.shape[0] - 1):
          # 30.4375 = (3 * 365 + 366) / 48.
          months = (all_fights.loc[i + 1, 'Date'] - all_fights.loc[i, 'Date']) / pd.to_timedelta('30.4375 days')
          months_12 = months
          months2 = int(months / 3.0)
          months = round(months)
          if (all_fights.loc[i + 1, 'Winner'] == fighter and all_fights.loc[i + 1, 'Outcome'] == 'def.'):
               wins[months] += 1
               wins2[months2] += 1
               if (months_12 > 15.0 and months_12 <= 24.0): wins_12 += 1
          total[months] += 1
          total2[months2] += 1
          if (months_12 > 15.0 and months_12 <= 24.0): total_12 += 1

ws = pd.Series(data=wins.values(), index=wins.keys())
ts = pd.Series(data=total.values(), index=total.keys())

df = pd.DataFrame([ws, ts]).T
df.columns = ['wins', 'total']
cowboy = df.copy()
df = df.loc[1:24]
df['WinRatio'] = df.wins / df.total
df['2se'] = 1.96 * np.sqrt(df.WinRatio * (1 - df.WinRatio) / df.total)
df

df.shape[0]

(797+1082+1014+719)/5783.

cowboy.sum()

cowboy

plt.bar(cowboy.index, cowboy.total)
plt.xlabel('Months Between Fights')
plt.ylabel('Count')
#plt.axes().minorticks_on()
minor_ticks = np.arange(0, 25, 1)
plt.axes().set_xticks(minor_ticks, minor = True)
plt.xlim(-0.5, 24.5)
plt.savefig('report/age/time_between_fights.pdf', bbox_inches='tight')

df.loc[15:25].sum(axis=0)

fig, ax = plt.subplots()
plt.plot([0, 25], [50, 50], 'k:')
plt.errorbar(df.index, 100 * df.WinRatio, color='k', marker='o', mfc='w', yerr=100*df['2se'], ecolor='gray', elinewidth=0.5, capsize=2)
#plt.plot(df.index, 100 * df.WinRatio, 'wo')
plt.xlabel('Months Since Last Fight')
plt.ylabel('Win Percentage')
plt.xlim(0, 25)
plt.ylim(0, 100)
major_ticks = np.arange(0, 28, 4)
ax.set_xticks(major_ticks)
minor_ticks = np.arange(0, 25, 1)
ax.set_xticks(minor_ticks, minor = True)
#plt.savefig('report/ring_rust.pdf', bbox_inches='tight')

ws = pd.Series(data=wins2.values(), index=wins2.keys())
ts = pd.Series(data=total2.values(), index=total2.keys())

df = pd.DataFrame([ws, ts]).T
df.columns = ['wins', 'total']
df = df.loc[0:4]
df.loc[5] = [wins_12, total_12]
df['WinRatio'] = df.wins / df.total
df['2se'] = -t.ppf(0.025, df.total - 1) * np.sqrt(df.WinRatio * (1 - df.WinRatio) / df.total)
df

df.index = [0, 1, 2, 3, 4, 5.5]

fig, ax = plt.subplots()
plt.plot([-1, 12], [50, 50], 'k:')
plt.errorbar(df.index, 100 * df.WinRatio, color='k', marker='o', mfc='w', yerr=100*df['2se'], ecolor='gray', elinewidth=0.5, capsize=2)
#plt.plot(df.index, 100 * df.WinRatio, 'wo')
plt.xlabel('Time Since Last Fight', labelpad=10)
plt.ylabel('Win Percentage')
plt.xlim(-.5, 6)
plt.ylim(30, 70)
major_ticks = [0, 1, 2, 3, 4, 5.5]
ax.set_xticks(major_ticks)
ax.set_xticklabels(['0 - 3\nMonths', '3 - 6\nMonths', '6 - 9\nMonths', '9 - 12\nMonths', '12 - 15\nMonths', '15 - 24\nMonths'])
#minor_ticks = np.arange(0, 25, 1)
#ax.set_xticks(minor_ticks, minor = True)
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#fig.subplots_adjust(bottom=0.0)
plt.savefig('report/age/ring_rust.pdf', bbox_inches='tight')

from scipy.stats import chi2_contingency

df['loses'] = df.total - df.wins
chi2, p, dof, expt = chi2_contingency(df[['wins', 'loses']])
chi2, p

df[['wins', 'loses']].T

expt.T

tmp_table = df[['wins', 'loses']]
N = tmp_table.sum().sum()
V = (chi2 / (N * min(tmp_table.shape[0] - 1, tmp_table.shape[1] - 1)))**0.5
V

chi2, p, dof, expt = chi2_contingency(df[['wins', 'total']])
chi2, p

df[['wins', 'total']].T

expt.T

