import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')
from scipy.stats import binom

fights = pd.read_csv('data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv', header=0, parse_dates=['Date'])
iofile = 'data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv'
fighters = pd.read_csv(iofile, header=0, parse_dates=['Dob'])
cols = ['Name', 'Height', 'Reach', 'LegReach', 'Stance', 'Dob']
df = fights.merge(fighters[cols], how='left', left_on='Winner', right_on='Name')
df = df.merge(fighters[cols], how='left', left_on='Loser', right_on='Name', suffixes=('', '_L'))
df.head(3)

msk = pd.notnull(df.Reach) & pd.notnull(df.Reach_L) & df.Outcome.isin(['def.', 'draw']) & (df.Reach != df.Reach_L)
af = df[msk]
total_fights = af.shape[0]
total_fights

af[af.Reach > af.Reach_L].shape[0]

af[af.Reach < af.Reach_L].shape[0]

2 * binom.cdf(p=0.5, k=1406, n=2952)

1546/2952., 1406/2952.

1406+1546

fig, ax = plt.subplots(figsize=(4, 3))
win_ratios = [100 * float(1546) / total_fights, 100 * float(1406) / total_fights]
plt.bar([0], win_ratios[0], width=0.5, align='center')
plt.bar([1], win_ratios[1], width=0.5, align='center')
plt.xlim(-0.5, 1.5)
plt.ylim(0, 100)
plt.xticks([0, 1])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Longer\nReach', 'Shorter\nReach'])
plt.ylabel('Win percentage')
plt.text(0, 55, '52.4%', ha='center')
plt.text(1, 50.5, '47.6%', ha='center')
plt.savefig('report/reach_height/longer_reach_win_ratio.pdf', bbox_inches='tight')

msk = pd.notnull(df.Reach) & pd.notnull(df.Reach_L) & pd.notnull(df.Height) & pd.notnull(df.Height_L) & df.Outcome.isin(['def.', 'draw']) & (df.Reach == df.Reach_L) & (df.Height != df.Height_L)
af = df[msk]
total_fights = af.shape[0]
total_fights

af[af.Height > af.Height_L].shape[0]

af[af.Height < af.Height_L].shape[0]

2 * binom.cdf(p=0.5, k=160, n=327)

msk = pd.notnull(df.Height) & pd.notnull(df.Height_L) & df.Outcome.isin(['def.', 'draw']) & (df.Height != df.Height_L)
af = df[msk]
total_fights = af.shape[0]
total_fights

af[af.Height > af.Height_L].shape[0]

af[af.Height < af.Height_L].shape[0]

2 * binom.cdf(p=0.5, k=1599, n=3282)

df['ReachDiff'] = df.Reach - df.Reach_L
df['ReachDiffAbs'] = np.abs(df.Reach - df.Reach_L)

win_ufc = df[['Winner', 'Reach', 'Height', 'LegReach']]
lose_ufc = df[['Loser', 'Reach_L', 'Height_L', 'LegReach_L']]
lose_ufc.columns = win_ufc.columns
win_lose_ufc = win_ufc.append(lose_ufc).drop_duplicates()
win_lose_ufc.columns = ['Name', 'Reach', 'Height', 'LegReach']
win_lose_ufc['Reach2Height'] = win_lose_ufc.Reach / win_lose_ufc.Height
win_lose_ufc.head(3)

win_lose_ufc.sort_values('Reach', ascending=False).head(3)

win_lose_ufc.shape[0]

tmp = win_lose_ufc[['Reach', 'Height']].dropna()
tmp.shape[0], tmp.shape[0] / 1641.

from scipy.stats import norm

dx = norm.rvs(loc=0.0, scale=0.1, size=pd.notnull(win_lose_ufc.Height).size)
dy = norm.rvs(loc=0.0, scale=0.1, size=pd.notnull(win_lose_ufc.Reach).size)

plt.plot(win_lose_ufc.Height + dx, win_lose_ufc.Reach + dy, 'wo')
plt.plot([55, 90], [55, 90], 'k-')
plt.xlabel('Height (inches)')
plt.ylabel('Reach (inches)')
plt.savefig('report/reach_height/height_reach_all_fighters.pdf', bbox_tight='inches')

from scipy.stats import pearsonr, spearmanr

hr = win_lose_ufc[['Height', 'Reach']].dropna()
corr_pearson, p_value_pearson = pearsonr(hr.Height, hr.Reach)
corr_spearman, p_value_spearman = spearmanr(hr.Height, hr.Reach)
print corr_pearson, p_value_pearson
print corr_spearman, p_value_spearman

tmp = win_lose_ufc[['Reach', 'LegReach']].dropna()

dx = norm.rvs(loc=0.0, scale=0.1, size=tmp.LegReach.size)
dy = norm.rvs(loc=0.0, scale=0.1, size=tmp.Reach.size)

plt.plot(tmp.LegReach + dx, tmp.Reach + dy, 'wo')
#plt.plot([55, 90], [55, 90], 'k-')
plt.xlabel('Leg Reach (inches)')
plt.ylabel('Reach (inches)')
plt.xlim(30, 50)
#plt.savefig('report/reach_height/leg_reach_all_fighters.pdf', bbox_tight='inches')

corr_pearson, p_value_pearson = pearsonr(tmp.LegReach, tmp.Reach)
corr_spearman, p_value_spearman = spearmanr(tmp.LegReach, tmp.Reach)
print corr_pearson, p_value_pearson
print corr_spearman, p_value_spearman

above9 = df.sort_values(['ReachDiffAbs', 'Date'], ascending=False)
above9 = above9[['Winner', 'Reach', 'Outcome', 'Loser', 'Reach_L', 'ReachDiffAbs', 'Date']][above9.ReachDiffAbs >= 9]
win_count = above9[above9.Reach > above9.Reach_L].shape[0]
above9 = above9.astype({'Reach':int, 'Reach_L':int, 'ReachDiffAbs':int})
above9.index = range(1, above9.shape[0] + 1)
above9.columns = ['Winner', 'Reach', 'Outcome', 'Loser', 'Reach', r'$\Delta$', 'Date']
above9.to_latex('report/reach_height/biggest_reach_diff_RAW.tex')
above9

cols = ['Name', 'Reach', 'Height', 'Reach2Height']
raw_reach = win_lose_ufc.sort_values(['Reach2Height'], ascending=False).reset_index(drop=True)[cols]
raw_reach = raw_reach[raw_reach.Reach2Height >= 1.08]
raw_reach = raw_reach.astype({'Reach':int, 'Height':int})
raw_reach.index = range(1, raw_reach.shape[0] + 1)
raw_reach.Reach2Height = raw_reach.Reach2Height.apply(lambda x: round(x, 2))
raw_reach.columns = ['Name', 'Reach', 'Height', 'Reach/Height']
raw_reach

cols = ['Name', 'Reach', 'Height', 'Reach2Height']
sm_reach = win_lose_ufc.sort_values(['Reach2Height'], ascending=True).reset_index(drop=True)[cols]
sm_reach = sm_reach[sm_reach.Reach2Height <= 0.98]
sm_reach = sm_reach.astype({'Reach':int, 'Height':int})
sm_reach.index = range(1, sm_reach.shape[0] + 1)
sm_reach.Reach2Height = sm_reach.Reach2Height.apply(lambda x: round(x, 2))
sm_reach.columns = ['Name', 'Reach', 'Height', 'Reach/Height']
sm_reach = sm_reach.loc[1:35]

# join the two tables
cmb = raw_reach.merge(sm_reach, left_index=True, right_index=True)
cmb.columns = ['Name', 'Reach', 'Height', 'Reach2Height', 'Name', 'Reach', 'Height', 'Reach2Height']
cmb.to_latex('report/reach_height/reach2height_large_RAW.tex')
cmb

win_count

df05 = df[(df.Date > pd.to_datetime('2005-01-01'))]

df05[df05.Loser == 'Naoyuki Kotani']

fighter_winratio = []
for fighter in df05.Winner.append(df05.Loser).unique():
     wins = df05[(df05.Winner == fighter) & (df05.Outcome == 'def.')].shape[0]
     loses = df05[(df05.Loser == fighter) & (df05.Outcome == 'def.')].shape[0]
     draws = df05[((df05.Winner == fighter) | (df05.Loser == fighter)) & (df05.Outcome == 'draw')].shape[0]
     total_fights = wins + loses + draws
     if total_fights > 4: fighter_winratio.append((fighter, (wins + 0.5 * draws) / total_fights))

fighter_winratio = pd.DataFrame(fighter_winratio, columns=['Name', 'WinRatio'])
fighter_winratio.head(3)

win_reach_ratios = fighter_winratio.merge(win_lose_ufc, on='Name', how='left')
win_reach_ratios = win_reach_ratios[pd.notnull(win_reach_ratios.Reach2Height)][['Name', 'WinRatio', 'Reach2Height']]
win_reach_ratios.head(3)

fighter_winratio[fighter_winratio.WinRatio < 0.1]

m, b = np.polyfit(win_reach_ratios.Reach2Height, 100 * win_reach_ratios.WinRatio, 1)
plt.plot(np.linspace(0.9, 1.15), m * np.linspace(0.9, 1.15) + b, 'k-')
plt.plot(win_reach_ratios.Reach2Height, 100 * win_reach_ratios.WinRatio, 'wo')
plt.xlabel('Reach-to-Height Ratio')
plt.ylabel('Win Percentage')
plt.xlim(0.9, 1.15)
plt.ylim(0, 110)
plt.savefig('report/reach_height/reach_vs_win_percent.pdf', bbox_tight='inches')

corr_pearson, p_value_pearson = pearsonr(win_reach_ratios.Reach2Height, win_reach_ratios.WinRatio)
corr_spearman, p_value_spearman = spearmanr(win_reach_ratios.Reach2Height, win_reach_ratios.WinRatio)
print corr_pearson, p_value_pearson
print corr_spearman, p_value_spearman

win_jones = df[['Winner', 'Reach', 'Height', 'WeightClass']]
lose_jones = df[['Loser', 'Reach_L', 'Height_L', 'WeightClass']]
lose_jones.columns = win_jones.columns
win_lose_jones = win_jones.append(lose_jones).drop_duplicates()
win_lose_jones = win_lose_jones[win_lose_jones.WeightClass == 'Light Heavyweight']
win_lose_jones.head(3)

win_lose_jones.shape[0]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

dx = norm.rvs(loc=0.0, scale=0.1, size=pd.notnull(win_lose_jones.Height).size)
dy = norm.rvs(loc=0.0, scale=0.1, size=pd.notnull(win_lose_jones.Reach).size)

ax1.plot(win_lose_jones.Height + dx, win_lose_jones.Reach + dy, 'wo')
ax1.set_xlim(66, 80)
ax1.set_ylim(68, 86)
ax1.arrow(73, 84, 2.0, 0, head_width=1, head_length=0.5, fc='k', ec='k')
ax1.text(71.1, 84, 'Jones', fontsize=12, va='center')
ax1.set_xlabel('Height (inches)')
ax1.set_ylabel('Reach (inches)')

_, _, patches = ax2.hist(win_lose_jones[pd.notnull(win_lose_jones.Reach)].Reach, bins=np.arange(68.5, 85.5, 1), color='lightgray')
patches[0].set_snap(True)
ax2.arrow(84, 9, 0, -5, head_width=0.7, head_length=1.4, fc='k', ec='k')
ax2.text(84, 11, 'Jones', fontsize=12, ha='center')
ax2.set_xlabel('Reach (inches)')
ax2.set_ylabel('Count')
#plt.tight_layout()
fig.savefig('report/reach_height/jones_reach.pdf', bbox_inches='tight')

df = df[(df.Date > pd.to_datetime('2005-01-01')) & (df.Outcome == 'def.')]

by_diff = df.ReachDiff.apply(lambda x: round(x)).value_counts().sort_index()
by_diff

by_diff_abs = df.ReachDiffAbs.apply(lambda x: round(x)).value_counts().sort_index()
by_diff_abs

from scipy.stats import t, norm

rdf = pd.DataFrame({'N':by_diff_abs.loc[1:10], 'Wins':by_diff.loc[1:10], 'WinRatio':by_diff.loc[1:10] / by_diff_abs.loc[1:10]})
rdf['Loses'] = rdf.N - rdf.Wins
rdf['2se_t'] = -t.ppf(0.025, rdf.N - 1) * np.sqrt(rdf.WinRatio * (1.0 - rdf.WinRatio) / rdf.N)
rdf['2se_z'] = -norm.ppf(0.025) * np.sqrt(rdf.WinRatio * (1.0 - rdf.WinRatio) / rdf.N)
rdf

cont_table = rdf[['Wins', 'Loses']].T
cont_table

from scipy.stats import chi2_contingency

chi_sq, p_value, dof, expect = chi2_contingency(cont_table)
print chi_sq, p_value, p_value > 0.05

N = cont_table.sum().sum()
V = (chi_sq / (N * min(cont_table.shape[0] - 1, cont_table.shape[1] - 1)))**0.5
V

fig, ax = plt.subplots()
plt.plot([0, 25], [50, 50], 'k:')
plt.errorbar(rdf.index, 100 * rdf.WinRatio, color='k', marker='o', mfc='w', yerr=100*rdf['2se_t'], ecolor='gray', elinewidth=0.5, capsize=2)
plt.xlabel('Reach Difference (inches)')
plt.ylabel('Win Percentage of\n Fighter with Longer Reach')
plt.xlim(0, 12)
plt.ylim(0, 100)
major_ticks = np.arange(0, 13, 1)
ax.set_xticks(major_ticks)
#minor_ticks = np.arange(0, 25, 1)
#ax.set_xticks(minor_ticks, minor = True)
plt.savefig('report/reach_height/winratio_reach_diff.pdf', bbox_inches='tight')

