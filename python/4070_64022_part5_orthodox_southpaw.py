import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

fights = pd.read_csv('data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv', header=0, parse_dates=['Date'])
iofile = 'data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv'
fighters = pd.read_csv(iofile, header=0, parse_dates=['Dob'])
fighters['Age'] = (pd.to_datetime('today') - fighters.Dob) / np.timedelta64(1, 'Y')
cols = ['Name', 'Height', 'Reach', 'Stance', 'Dob', 'Age']
df = fights.merge(fighters[cols], how='left', left_on='Winner', right_on='Name')
df['AgeThen'] = (df.Date - df.Dob) / np.timedelta64(1, 'Y')
df = df.merge(fighters[cols], how='left', left_on='Loser', right_on='Name', suffixes=('', '_L'))
df['AgeThen_L'] = (df.Date - df.Dob_L) / np.timedelta64(1, 'Y')

win_lose = fights.Winner.append(fights.Loser).unique()
win_lose = pd.DataFrame(win_lose, columns=['Name'])
win_lose = win_lose.merge(fighters, on='Name', how='left')
win_lose.Stance.value_counts()

stance_overview = pd.DataFrame([win_lose.Stance.value_counts(normalize=False), 100 * win_lose.Stance.value_counts(normalize=True)]).T.applymap(lambda x: round(x, 1))
stance_overview.columns = ['Count', 'Percentage']
stance_overview = stance_overview.astype({'Count':int})
stance_overview.T.to_latex('report/stance_breakdown_RAW.tex')

ortho_south = df[(df.Outcome.isin(['def.'])) & (df.Date > np.datetime64('2005-01-01'))].copy()
msk1 = ((ortho_south.Stance == 'Orthodox') & (ortho_south.Stance_L == 'Southpaw'))
msk2 = ((ortho_south.Stance == 'Southpaw') & (ortho_south.Stance_L == 'Orthodox'))
cols = ['Winner', 'Outcome', 'Loser', 'Stance', 'Stance_L', 'Reach', 'Reach_L', 'Age', 'Age_L', 'Date', 'AgeThen', 'AgeThen_L']
ortho_south = ortho_south[msk1 | msk2][cols]
cols = ['Winner', 'Stance', 'Loser', 'Stance_L', 'Date']
top25 = ortho_south.sort_values('Date', ascending=False).reset_index(drop=True)[cols]
top25.index = range(1, top25.shape[0] + 1)
top25.columns = ['Winner', 'Stance', 'Loser', 'Stance', 'Date']
top25.to_latex('report/southpaw/ortho_vs_south_RAW.tex')
top25

total_fights = ortho_south.shape[0]
total_fights

unique_fighters = ortho_south.Winner.append(ortho_south.Loser).unique()
unique_fighters.size

cont_table = fighters[fighters.Name.isin(unique_fighters)].groupby('Stance').agg({'Height':[np.size, np.mean, np.std], 'Reach':[np.mean, np.std], 'Age':[np.mean, np.std]})
cont_table.astype({('Height', 'size'):int}).applymap(lambda x: round(x, 3))

w_ortho = ortho_south[ortho_south.Stance == 'Orthodox'].shape[0]
w_south = ortho_south[ortho_south.Stance == 'Southpaw'].shape[0]
l_ortho = ortho_south[ortho_south.Stance_L == 'Orthodox'].shape[0]
l_south = ortho_south[ortho_south.Stance_L == 'Southpaw'].shape[0]

cont_table = pd.DataFrame([[w_ortho, w_south], [l_ortho, l_south]])
cont_table.columns = ['Orthodox', 'Southpaw']
cont_table.index=['Wins', 'Losses']
cont_table

cont_table / cont_table.sum(axis=0)

from scipy.stats import chisquare

chi_sq_stat, p_value = chisquare([w_ortho, w_south], [0.5 * total_fights, 0.5 * total_fights])
chi_sq_stat, p_value

from scipy.stats import binom
2 * sum([binom.pmf(k, n=total_fights, p=0.5) for k in range(0, 454 + 1)])

p_value = 2 * binom.cdf(k=454, n=total_fights, p=0.5)
p_value

2 * (1.0 - binom.cdf(k=562, n=total_fights, p=0.5))

win_ratio = [float(w_ortho) / total_fights, float(w_south) / total_fights]
plt.bar(range(cont_table.shape[1]), win_ratio, width=0.5, tick_label=cont_table.columns, align='center')
plt.ylim(0, 1)
plt.ylabel('Win ratio')

stance_reach = ortho_south.copy()

stance_reach['ReachDiff'] = np.abs(stance_reach.Reach - stance_reach.Reach_L)
stance_reach['AgeDiff'] = np.abs(stance_reach.Age - stance_reach.Age_L)
stance_reach = stance_reach[(stance_reach.ReachDiff <= 3.0) & (stance_reach.AgeDiff <= 3.0)]

w_ortho = stance_reach[stance_reach.Stance == 'Orthodox'].shape[0]
w_south = stance_reach[stance_reach.Stance == 'Southpaw'].shape[0]
l_ortho = stance_reach[stance_reach.Stance_L == 'Orthodox'].shape[0]
l_south = stance_reach[stance_reach.Stance_L == 'Southpaw'].shape[0]

cols = ['Winner', 'Stance', 'AgeThen', 'Reach', 'Loser', 'Stance_L', 'AgeThen_L', 'Reach_L', 'Date']
top25 = stance_reach.sort_values('Date', ascending=False).reset_index(drop=True)[cols]
top25.AgeThen = top25.AgeThen.apply(lambda x: round(x, 1))
top25.AgeThen_L = top25.AgeThen_L.apply(lambda x: round(x, 1))
top25.Reach = top25.Reach.astype(int)
top25.Reach_L = top25.Reach_L.astype(int)
top25.index = range(1, top25.shape[0] + 1)
top25.columns = ['Winner', 'Stance', 'Age', 'Reach', 'Loser', 'Stance','Age','Reach', 'Date']
top25.to_latex('report/southpaw/ortho_vs_south_same_age_reach_RAW.tex')
top25

total_fights = stance_reach.shape[0]
total_fights

unique_fighters = stance_reach.Winner.append(stance_reach.Loser).unique()
unique_fighters.size

cont_table = pd.DataFrame([[w_ortho, w_south], [l_ortho, l_south]])
cont_table.columns = ['Orthodox', 'Southpaw']
cont_table.index=['Wins', 'Losses']
cont_table

cont_table / cont_table.sum(axis=0)

fig, ax = plt.subplots(figsize=(4, 3))
win_ratios = [100 * float(w_ortho) / total_fights, 100 * float(w_south) / total_fights]
plt.bar([0], win_ratios[1], width=0.5, align='center')
plt.bar([1], win_ratios[0], width=0.5, align='center')
plt.xlim(-0.5, 1.5)
plt.ylim(0, 100)
plt.xticks([0, 1])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Southpaw', 'Orthodox'])
plt.ylabel('Win percentage')
plt.text(1, 45, '41.8%', ha='center')
plt.text(0, 61, '58.2%', ha='center')
plt.savefig('report/southpaw/southpaw_win_ratio.pdf', bbox_inches='tight')

chi_sq, p_value = chisquare(cont_table.loc['Wins'])
print chi_sq, p_value, p_value > 0.05

p_value = 2 * binom.cdf(k=w_ortho, n=total_fights, p=0.5)
p_value

with open('data/ranked_ufc_fighters_1488838405.txt') as f:
     ranked = f.readlines()
ranked = [fighter.strip() for fighter in ranked]

rf = pd.DataFrame(ranked)
rf.columns = ['Name']
rf['Ranked'] = 1

af = pd.read_csv('data/weight_class_majority.csv', header=0)
ranked_active = rf.merge(af[af.Active == 1], on='Name', how='right')
ranked_active.head(3)

stance_ranked = fighters.merge(ranked_active, on='Name', how='right')
stance_ranked.head(3)[['Name', 'Ranked', 'Stance']]

stance_ranked.shape[0]

overall = stance_ranked.Stance.value_counts()
overall

overall['Southpaw'] / float(overall.sum())

among_ranked = stance_ranked[pd.notnull(stance_ranked.Ranked)].Stance.value_counts()
among_ranked

among_ranked['Southpaw'] / float(among_ranked.sum())

df['Year'] = df.Date.dt.year
w_stance_year = df[['Stance', 'Year']]
l_stance_year = df[['Stance_L', 'Year']]
l_stance_year.columns = ['Stance', 'Year']
cmb = w_stance_year.append(l_stance_year)

year_stance = pd.crosstab(index=cmb["Year"], columns=cmb["Stance"])
year_stance['Total'] = year_stance.sum(axis=1) # compute total before other
year_stance['Other'] = year_stance['Open Stance'] + year_stance['Sideways'] + year_stance['Switch']
year_stance

year_stance.Total.sum()

df[['Stance', 'Stance_L']].info()

clrs = plt.rcParams['axes.prop_cycle']
clrs = [color.values()[0] for color in list(clrs)]

plt.plot(year_stance.index, 100 * year_stance.Orthodox / year_stance.Total, '-', marker='o', mfc='w', mec=clrs[0], label='Orthodox')
plt.plot(year_stance.index, 100 * year_stance.Southpaw / year_stance.Total, '-', marker='o', mfc='w', mec=clrs[1], label='Southpaw')
plt.plot(year_stance.index, 100 * year_stance.Other / year_stance.Total, '-', marker='o', mfc='w', mec=clrs[2], label='Other')
plt.ylim(0, 100)
plt.xlabel('Year')
plt.ylabel('Stance (%)')
plt.legend(loc=(0.65, 0.35), fontsize=11, markerscale=1)
plt.savefig('report/southpaw/stance_type_by_year.pdf', bbox_inches='tight')

x = 100 * year_stance.Southpaw / year_stance.Total
x.loc[2005:2016].mean()

