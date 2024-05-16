import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('halverson')
get_ipython().magic('matplotlib inline')

df = pd.read_csv('data/fightmetric_fighters/fightmetric_fighters_CLEAN_3-6-2017.csv', header=0, parse_dates=['Dob'])
df['Age'] = (pd.to_datetime('today') - df.Dob) / np.timedelta64(1, 'Y')
df.Age = df.Age.apply(lambda x: round(x, 1))
#pd.set_option('display.max_rows', 3000)
df.head(10)

df.info()

df.shape[0]

df.describe().applymap(lambda x: round(x, 2))

min(df.Dob), max(df.Dob)

df[pd.notnull(df.Dob)].Dob.apply(lambda x: (x.month, x.day)).value_counts()[:5]

df[(df.Dob.dt.month == 2) & (df.Dob.dt.day == 29)]

bd_counts = df[pd.notnull(df.Dob)].Dob.dt.month.value_counts()
plt.bar(bd_counts.index, bd_counts.values, align='center')
plt.xlim(0, 13)
plt.xlabel('Month of Year')
plt.ylabel('Count')

df[pd.notnull(df.Dob)].sort_values('Dob', ascending=False).head(5)

name_counts = df.Name.value_counts()
name_counts[name_counts > 1]

df[(df.Name == 'Michael McDonald') | (df.Name == 'Tony Johnson') | (df.Name == 'Dong Hyun Kim')]

# rename the second instance
idx = df[(df.Name == 'Tony Johnson') & (df.Weight == 265)].index.values
df = df.set_value(idx, 'Name', 'Tony Johnson 2')

# rename the second instance
idx = df[(df.Name == 'Dong Hyun Kim') & (df.Nickname == 'Maestro')].index.values
df = df.set_value(idx, 'Name', 'Dong Hyun Kim 2')

# rename the second instance
idx = df[(df.Name == 'Michael McDonald') & (df.Nickname == 'The Black Sniper')].index.values
df = df.set_value(idx, 'Name', 'Michael McDonald 2')

name_counts = df.Name.value_counts()
name_counts[name_counts > 1]

df[df.Name.str.contains('  ')][['Name']]

df[df.Name.apply(lambda x: not ''.join(x.split()).isalpha())][['Name']]

pd.set_option('display.max_rows', 100)
df[df.Name.apply(lambda x: len(x.split()) != 2)][['Name']]

plt.plot(df.Height, df.Weight, 'wo')
plt.xlabel('Heights (inches)')
plt.ylabel('Weight (lbs.)')

df.dropna(subset=['Weight']).sort_values('Weight', ascending=False).head(5)

df.dropna(subset=['Height']).sort_values('Height', ascending=False).head(5)

counts, edges, patches = plt.hist(df.Reach.dropna(), bins=np.arange(55.5, 90.5, 1.0))
patches[0].set_snap(True)
plt.xlabel('Reach (inches)')
plt.ylabel('Count')

df[pd.notnull(df.Reach)].sort_values('Reach', ascending=False).head(5)

plt.plot([55, 90], [55, 90], 'k-')
plt.plot(df.Height, df.Reach, 'wo')
plt.xlabel('Height (inches)')
plt.ylabel('Reach (inches)')
plt.xlim(55, 90)
plt.ylim(55, 90)

today = pd.to_datetime('today').to_pydatetime()
date = '-'.join(map(str, [today.month, today.day, today.year]))
cols = ['Name', 'Nickname', 'Dob', 'Age', 'Weight', 'Height', 'Reach', 'Stance', 'Win', 'Loss', 'Draw']
df[cols].to_csv('data/fightmetric_fighters/fightmetric_fighters_CLEAN_' + date + '.csv', index=False)

df['ReachHeight'] = df.Reach / df.Height
df.drop(['Nickname', 'Name'], axis=1).dropna(subset=['Reach', 'Height']).sort_values('ReachHeight', ascending=False).head(10)

df.drop(['Nickname', 'Name'], axis=1).dropna(subset=['Reach', 'Height']).sort_values('ReachHeight', ascending=True).head(10)

df['Fights'] = df['Win'] + df['Loss'] + df['Draw']
df['WinRatio'] = df['Win'] / df['Fights']

df.sort_values('Fights', ascending=False).head(5).drop(['Nickname', 'Name'], axis=1)

df[df.Fights > 15].sort_values('WinRatio', ascending=False).head(5).drop(['Nickname', 'Name'], axis=1)

plt.hist(df.WinRatio.dropna(), bins=25)
plt.xlabel('Win ratio')
plt.ylabel('Count')

f10 = df[df.Fights > 10][['WinRatio', 'ReachHeight']].dropna()
m, b = np.polyfit(f10.ReachHeight.values, f10.WinRatio.values, 1)
plt.plot(f10.ReachHeight, f10.WinRatio, 'wo')
plt.plot(np.linspace(0.9, 1.15), m * np.linspace(0.9, 1.15) + b, 'k-')
plt.xlim(0.9, 1.15)
plt.ylim(0, 1.2)
plt.xlabel('Reach / Height')
plt.ylabel('Win ratio')

from scipy.stats import pearsonr, spearmanr

corr_pearson, p_value_pearson = pearsonr(f10.ReachHeight, f10.WinRatio)
corr_spearman, p_value_spearman = spearmanr(f10.ReachHeight, f10.WinRatio)
print corr_pearson, p_value_pearson
print corr_spearman, p_value_spearman

stance_overview = pd.DataFrame([df.Stance.value_counts(normalize=False), 100 * df.Stance.value_counts(normalize=True)]).T.applymap(lambda x: round(x, 2))
stance_overview.columns = ['Count', 'Percentage']
stance_overview.astype({'Count':int})

df.groupby('Stance').agg({'WinRatio':[np.size, np.mean, np.std], 'Height':np.mean, 'Reach':np.mean})

f10_stance = df[df.Stance.isin(['Orthodox', 'Southpaw']) & (df.Fights > 10)]
stance = f10_stance.groupby('Stance').agg({'WinRatio':[np.mean, np.std], 'Height':[np.size, np.mean, np.std], 'Reach':[np.mean, np.std]})
stance.astype({('Height', 'size'):int}).applymap(lambda x: round(x, 3))

fig = plt.figure(1, figsize=(4, 3))
plt.bar(range(stance.shape[0]), stance[('WinRatio', 'mean')], width=0.5, tick_label=stance.index.values, align='center')
plt.ylim(0, 1)
plt.ylabel('Win ratio')

orthodox = f10_stance[(f10_stance.Stance == 'Orthodox')].WinRatio
southpaw = f10_stance[(f10_stance.Stance == 'Southpaw')].WinRatio

fig = plt.figure(1, figsize=(5, 4))
plt.boxplot([orthodox, southpaw], labels=['Orthodox', 'Southpaw'])
plt.ylabel('Win ratio')
plt.ylim(0, 1.2)

from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(orthodox, southpaw, equal_var=False)
print t_stat, p_value

row_orthodox = df[(df.Stance == 'Orthodox') & (df.Fights > 10)][['Win', 'Loss']].sum()
row_southpaw = df[(df.Stance == 'Southpaw') & (df.Fights > 10)][['Win', 'Loss']].sum()

cont_table = pd.DataFrame([row_orthodox, row_southpaw], index=['Orthodox', 'Southpaw']).T
cont_table['Total'] = cont_table.sum(axis=1)
cont_table.loc['Total'] = cont_table.sum(axis=0)
cont_table

cont_table.loc['Win', 'Orthodox'] / cont_table.loc['Total', 'Orthodox']

cont_table.loc['Win', 'Southpaw'] / cont_table.loc['Total', 'Southpaw']

win_ortho_expect = cont_table.loc['Win', 'Total'] * cont_table.loc['Total', 'Orthodox'] / cont_table.loc['Total', 'Total']
win_south_expect = cont_table.loc['Win', 'Total'] * cont_table.loc['Total', 'Southpaw'] / cont_table.loc['Total', 'Total']
los_ortho_expect = cont_table.loc['Loss', 'Total'] * cont_table.loc['Total', 'Orthodox'] / cont_table.loc['Total', 'Total']
los_south_expect = cont_table.loc['Loss', 'Total'] * cont_table.loc['Total', 'Southpaw'] / cont_table.loc['Total', 'Total']
expect = pd.DataFrame([[win_ortho_expect, win_south_expect], [los_ortho_expect, los_south_expect]], index=['Win', 'Loss'], columns=['Orthodox', 'Southpaw'])
expect

from scipy.stats import chi2

chi_sq = cont_table.iloc[0:2, 0:2].subtract(expect).pow(2).divide(expect).values.sum()
p_value = 1.0 - chi2.cdf(chi_sq, df=(2 - 1) * (2 - 1))
print chi_sq, p_value, p_value > 0.05

from scipy.stats import chi2_contingency

chi_sq, p_value, dof, expect = chi2_contingency(cont_table.iloc[0:2, 0:2].values, correction=False)
print chi_sq, p_value, p_value > 0.05

