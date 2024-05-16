import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')
from scipy.stats import binom, norm

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)

iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)

fights.shape

win_lose = fights.Winner.append(fights.Loser)
cts = win_lose.value_counts()
plt.hist(cts, bins=np.arange(0.5, 30.5, 1.0), rwidth=0.8, log=True)
plt.xlabel('Number of Fights')
plt.ylabel('Number of Fighters')

0.5 * cts.sum()

np.percentile(cts.values, 50)

plt.hist(cts, bins=np.arange(0.5, 30.5, 1.0), rwidth=0.8, cumulative=True, normed=True)
plt.xlabel('Number of Fights')
plt.ylabel('Number of Fighters')
plt.ylim(0, 1)

# below we use the index to find previous fights since in early days they fought
# multiple times per day so date cannot be used

NumPreviousFights = []
NumPreviousFights_L = []
for index, row in fights.iterrows():
     d = row['Date']
     
     winner = row['Winner']
     a = fights[((fights.Winner == winner) | (fights.Loser == winner)) & (fights.index > index)]
     NumPreviousFights.append(a.shape[0])
     
     loser = row['Loser']
     b = fights[((fights.Winner == loser) | (fights.Loser == loser)) & (fights.index > index)]
     NumPreviousFights_L.append(b.shape[0])
fights['NumPreviousFights'] = NumPreviousFights
fights['NumPreviousFights_L'] = NumPreviousFights_L

fights

f05 = fights[fights.Date > pd.to_datetime('2005-01-01')]

min_num_fights = []
min_num_fights05 = []
total_fights = float(fights.shape[0])
total_fights05 = float(f05.shape[0])
for i in range(25+1):
     min_num_fights.append((i, 100.0 * fights[(fights.NumPreviousFights <= i) | (fights.NumPreviousFights_L <= i)].shape[0] / total_fights))
     min_num_fights05.append((i, 100.0 * f05[(f05.NumPreviousFights <= i) | (f05.NumPreviousFights_L <= i)].shape[0] / total_fights05))
mins, ct = zip(*min_num_fights)
mins05, ct05 = zip(*min_num_fights05)
plt.plot(mins, ct, label='All Time')
plt.plot(mins05, ct05, label='Since 2005')
plt.xlabel('$m$')
plt.ylabel('Percentage of fights where one or\nboth fighters have $m$ fights or less')
plt.xlim(0, 15)
plt.ylim(0, 100)
plt.axes().set_xticks(range(0, 21, 2))
#plt.axes().grid(True)
print min_num_fights,'\n\n', min_num_fights05
plt.legend()
plt.savefig('report/prediction/lack_of_ufc_fights.pdf', bbox_inches='tight')

fights[((fights.NumPreviousFights >= 8) & (fights.NumPreviousFights_L >= 8)) & (fights.Date > pd.to_datetime('2005-01-01')) & (fights.Outcome == 'def.')].shape[0]

fights[(fights.NumPreviousFights >= 8) & (fights.NumPreviousFights_L >= 8)].shape[0]

