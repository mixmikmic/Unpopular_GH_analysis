import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')
from scipy.stats import binom, t
from scipy.stats import chi2_contingency

iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)

fights.shape

iofile = 'data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv'
fighters = pd.read_csv(iofile, header=0, parse_dates=['Dob'])
fighters.head(3)

cols = ['Name', 'Weight', 'Height', 'Reach', 'LegReach', 'Stance', 'Dob']
df = fights.merge(fighters[cols], how='left', left_on='Winner', right_on='Name')
df.shape

df.head(3)

df = df.merge(fighters[cols], how='left', left_on='Loser', right_on='Name', suffixes=('', '_L'))
df.shape

df.head(3)

df = df.drop(['Name', 'Name_L'], axis=1)

df.info()

df[(pd.isnull(df.Dob)) | (pd.isnull(df.Dob_L))].shape[0]

df['AgeDiffAbs'] = np.abs((df.Dob - df.Dob_L) / np.timedelta64(1, 'Y'))
df['Age'] = (df.Date - df.Dob) / np.timedelta64(1, 'Y')
df['Age_L'] = (df.Date - df.Dob_L) / np.timedelta64(1, 'Y')
cols = ['Winner','Age', 'Outcome', 'Loser','Age_L', 'AgeDiffAbs', 'Date']
big_age_diff = df.sort_values('AgeDiffAbs', ascending=False)[cols].reset_index(drop=True).head(40)
big_age_diff.AgeDiffAbs = big_age_diff.AgeDiffAbs.apply(lambda x: round(x, 1))
big_age_diff.Age = big_age_diff.Age.apply(lambda x: round(x, 1))
big_age_diff.Age_L = big_age_diff.Age_L.apply(lambda x: round(x, 1))
big_age_diff.index = range(1, big_age_diff.shape[0] + 1)
big_age_diff.to_latex('report/age/biggest_age_diff_RAW.tex')
big_age_diff

yw = big_age_diff[big_age_diff.Age < big_age_diff.Age_L].shape[0]
yw

total = big_age_diff.shape[0]
float(yw) / total

2*binom.cdf(p=0.5, k=min(yw, total - yw), n=total)

df[df.Dob == df.Dob_L]

#& (df.Date > np.datetime64('2005-01-01'))
msk = pd.notnull(df.Dob) & pd.notnull(df.Dob_L) & df.Outcome.isin(['def.', 'draw']) & (df.Dob != df.Dob_L)
af = df[msk]
total_fights = af.shape[0]
total_fights

winner_is_younger = float(af[(af.Dob > af.Dob_L) & (af.Outcome == 'def.')].shape[0]) / total_fights
winner_is_younger

af[(af.Dob > af.Dob_L) & (af.Outcome == 'def.')].shape[0]

winner_is_older = float(af[(af.Dob < af.Dob_L) & (af.Outcome == 'def.')].shape[0]) / total_fights
winner_is_older

af[(af.Dob < af.Dob_L) & (af.Outcome == 'def.')].shape[0]

other = float(af[af.Outcome == 'draw'].shape[0]) / total_fights
other

af[af.Outcome == 'draw'].shape[0]

winner_is_younger + winner_is_older + other

fig, ax = plt.subplots(figsize=(4, 3))
plt.bar([0], 100 * np.array([winner_is_younger, winner_is_older])[0], width=0.5, align='center')
plt.bar([1], 100 * np.array([winner_is_younger, winner_is_older])[1], width=0.5, align='center')
plt.xlim(-0.5, 1.5)
plt.ylim(0, 100)
plt.ylabel('Win percentage')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Younger\nFighter', 'Older\nFighter'])
w_pct_str = '%.1f' % (100 * winner_is_younger)
l_pct_str = '%.1f' % (100 * winner_is_older)
plt.text(1, 47, l_pct_str + '%', ha='center')
plt.text(0, 57, w_pct_str + '%', ha='center')
plt.savefig('report/age/win_pct_younger_older.pdf', bbox_inches='tight')

af[(af.Dob < af.Dob_L) & (af.Outcome == 'def.')].shape[0]

draws = af[af.Outcome == 'draw'].shape[0]
draws

p = 2 * binom.cdf(p=0.5, k=1716, n=3850 - 26)
p

df.Method.value_counts()

msk = pd.notnull(df.Dob) & pd.notnull(df.Dob_L) & df.Outcome.isin(['def.', 'draw']) & (df.Dob != df.Dob_L) & (df.Method.str.contains('DEC'))
dec = df[msk]
total_fights_dec = dec.shape[0]
total_fights_dec

winner_is_younger_dec = float(dec[(dec.Dob > dec.Dob_L) & (dec.Outcome == 'def.')].shape[0]) / total_fights_dec
winner_is_younger_dec

winner_is_older_dec = float(dec[(dec.Dob < dec.Dob_L) & (dec.Outcome == 'def.')].shape[0]) / total_fights_dec
winner_is_older_dec

bk = df[df.Outcome.isin(['def.', 'draw']) & (df.Date > np.datetime64('2005-01-01')) & pd.notnull(df.Dob) & pd.notnull(df.Dob_L)].copy()
#bk['Age'] = (bk.Date - bk.Dob) / np.timedelta64(1, 'Y')
#bk['Age_L'] = (bk.Date - bk.Dob_L) / np.timedelta64(1, 'Y')
bk['Age_int'] = bk.Age.apply(lambda x: round(x)).astype(int)
bk['Age_L_int'] = bk.Age_L.apply(lambda x: round(x)).astype(int)

bk.shape[0]

results = []
brackets = [(18, 24), (25, 29), (30, 34), (35, 39)]
for b_low, b_high in brackets:
     msk = (bk.Age_int <= b_high) & (bk.Age_int >= b_low) & (bk.Age_L_int <= b_high) & (bk.Age_L_int >= b_low)
     younger = bk[(bk.Age_int < bk.Age_L_int) & (bk.Outcome == 'def.') & msk].shape[0]
     older = bk[(bk.Age_int > bk.Age_L_int) & (bk.Outcome == 'def.') & msk].shape[0]
     same = bk[(bk.Age_int == bk.Age_L_int) & (bk.Outcome == 'def.') & msk].shape[0]
     total = float(bk[msk].shape[0])
     total_same = float(bk[(bk.Age_int == bk.Age_L_int) & msk].shape[0])
     results.append(( b_low, b_high, younger, older, younger / (total - total_same), same / (2 * total_same), older / (total - total_same), total, total_same))

def make_label(x):
     return str(int(x[0])) + '-' + str(int(x[1]))

results = pd.DataFrame(results, columns = ['b_low', 'b_high','count_younger','count_older', 'young', 'same', 'old', 'total', 'total_same'])
results['labels_'] = results.apply(make_label, axis=1)
results

627-143, 525-120, 67-16

2*binom.cdf(p=0.5, k=221, n=221+260)

2*binom.cdf(p=0.5, k=186, n=218+186)

2*binom.cdf(p=0.5, k=24, n=24+27)

fig, ax = plt.subplots()
left = np.arange(4)
plt.bar(left, 100*results.young, width=0.2, label='Younger')
plt.bar(left + 0.2, 100*results.same, width=0.2, label='Same Age')
plt.bar(left + 0.4, 100*results.old, width=0.2, label='Older')
plt.xlabel('Age Bracket')
plt.ylabel('Win Percentage')
plt.xlim(-0.2, 3.8)
plt.ylim(0, 80)
plt.legend(fontsize=11)
ax.set_xticks(left + 0.15)
ax.set_xticklabels(results.labels_)
ax.xaxis.set_ticks_position('none') 
#plt.savefig('report/age/age_brackets.pdf', bbox_inches='tight')

fig, ax = plt.subplots()
left = np.arange(4)
plt.bar(left, 100*results.young, width=0.3, label='Younger')
plt.bar(left + 0.3, 100*results.old, width=0.3, label='Older')
plt.xlabel('Age Bracket')
plt.ylabel('Win Percentage')
plt.xlim(-0.4, 3.7)
plt.ylim(0, 80)
plt.legend(fontsize=11)
ax.set_xticks(left + 0.15)
ax.set_xticklabels(results.labels_)
ax.xaxis.set_ticks_position('none') 
plt.savefig('report/age/age_brackets.pdf', bbox_inches='tight')

df.Method.value_counts()

# remove draws
bk = bk[bk.Outcome == 'def.']

results = []
brackets = [(18, 24), (25, 29), (30, 34), (35, 39)]
for b_low, b_high in brackets:
     msk = (bk.Age_int <= b_high) & (bk.Age_int >= b_low)
     dq = bk[(bk.Method == 'DQ') & msk].shape[0]
     sub = bk[(bk.Method == 'SUB') & msk].shape[0]
     tko = bk[(bk.Method == 'KO/TKO') & msk].shape[0]
     dec = bk[bk.Method.str.contains('DEC') & msk].shape[0]
     total = bk[msk].shape[0]
     results.append((b_low, b_high, tko, sub, dec, dq, total))

results = pd.DataFrame(results, columns = ['b_low', 'b_high', 'tko', 'sub', 'dec', 'dq', 'total'])
results['labels_'] = results.apply(make_label, axis=1)
results

tmp_table = results.loc[:, 'tko':'dec']
chi_sq, p_value, dof, expect = chi2_contingency(tmp_table)
print chi_sq, dof, p_value, p_value > 0.05

N = tmp_table.sum().sum()
V = (chi_sq / (N * min(tmp_table.shape[0] - 1, tmp_table.shape[1] - 1)))**0.5
V

results.loc[:, 'tko':'dq'].divide(results.total, axis=0)

cont_table = 100 * results.loc[:, 'tko':'dq'].divide(results.total, axis=0).applymap(lambda x: round(x, 3))
cont_table = cont_table.astype(str).applymap(lambda x: x + '%')
cont_table.columns = ['KO/TKO', 'Submission', 'Decision', 'Opponent DQ']
cont_table.index = results.labels_.values
cont_table.to_latex('report/age/finishes_by_age_RAW.tex')
cont_table

results.loc[:, 'tko':'dq'].divide(results.total, axis=0).sum(axis=1)

wins = df[df.Outcome.isin(['def.']) & (df.Date > np.datetime64('2005-01-01')) & pd.notnull(df.Dob) & pd.notnull(df.Dob_L)].copy()
#wins['Age'] = (wins.Date - wins.Dob) / np.timedelta64(1, 'Y')
#wins['Age_L'] = (wins.Date - wins.Dob_L) / np.timedelta64(1, 'Y')
wins['Age_int'] = wins.Age.apply(lambda x: round(x)).astype(int)
wins['Age_L_int'] = wins.Age_L.apply(lambda x: round(x)).astype(int)

msk1 = wins.Age < 25
msk2 = wins.Age_L < 25
under25 = float(wins[msk1].shape[0]) / wins[msk1 | msk2].shape[0]
under25

wins[msk1].shape[0], wins[msk2].shape[0]

2 * binom.cdf(p=0.5, n=wins[msk1].shape[0] + wins[msk2].shape[0], k=min(wins[msk1].shape[0], wins[msk2].shape[0]))

msk1 = (wins.Age >= 25) & (wins.Age <= 29)
msk2 = (wins.Age_L >= 25) & (wins.Age_L <= 29)
over25under30 = float(wins[msk1].shape[0]) / wins[msk1 | msk2].shape[0]
over25under30

wins[msk1].shape[0], wins[msk2].shape[0]

2 * binom.cdf(p=0.5, n=wins[msk1].shape[0] + wins[msk2].shape[0], k=min(wins[msk1].shape[0], wins[msk2].shape[0]))

msk1 = (wins.Age >= 30) & (wins.Age < 35)
msk2 = (wins.Age_L >= 30) & (wins.Age_L < 35)
over30under35 = float(wins[msk1].shape[0]) / wins[msk1 | msk2].shape[0]
over30under35

wins[msk1].shape[0], wins[msk2].shape[0]

2 * binom.cdf(p=0.5, n=wins[msk1].shape[0] + wins[msk2].shape[0], k=min(wins[msk1].shape[0], wins[msk2].shape[0]))

msk1 = wins.Age >= 35
msk2 = wins.Age_L >= 35
over35 = float(wins[msk1].shape[0]) / wins[msk1 | msk2].shape[0]
over35

wins[msk1].shape[0], wins[msk2].shape[0]

2 * binom.cdf(p=0.5, n=wins[msk1].shape[0] + wins[msk2].shape[0], k=min(wins[msk1].shape[0], wins[msk2].shape[0]))

wins[msk1 & msk2][['Winner', 'Loser', 'Age', 'Age_L']].shape[0]

msk1 = (wins.Age > 35) & (wins.Age_L < 35)
msk2 = (wins.Age_L > 35) & (wins.Age < 35)
over35 = float(wins[msk1].shape[0]) / wins[msk1 | msk2].shape[0]
over35

wins[msk1].shape[0], wins[msk2].shape[0]

win_percent = [under25, over25under30, over30under35, over35]
labels = ['18-24', '25-29', '30-34', '35 and above']
plt.plot([-1, 4], [50, 50], 'k:', zorder=0)
plt.bar(range(len(win_percent)), 100 * np.array(win_percent), color='lightgray', tick_label=labels, align='center')
plt.xlim(-0.6, 3.6)
plt.ylim(0, 100)
plt.ylabel('Win percentage')
plt.xlabel('Age Bracket')

wins

bounds = [(i, i + 2) for i in range(20, 40, 2)]
counts = []
for age_low, age_high in bounds:
     msk1 = ((wins.Age > age_low) & (wins.Age <= age_high))
     msk2 = ((wins.Age_L > age_low) & (wins.Age_L <= age_high))
     ct = wins[msk1 | msk2].shape[0]
     counts.append((age_low, age_high, wins[msk1].shape[0], ct))
cmb = pd.DataFrame(counts)
cmb.columns = ['low', 'high', 'wins', 'total']
cmb['WinRatio'] = cmb.wins / cmb.total
cmb['2se'] = -t.ppf(0.025, cmb.total - 1) * (cmb.WinRatio * (1.0 - cmb.WinRatio) / cmb.total)**0.5
cmb

x = cmb.low + 1
y = 100 * cmb.WinRatio.values
xmin = 20
xmax = 40

fig, ax = plt.subplots()
plt.plot([xmin, xmax], [50, 50], 'k:')
plt.errorbar(x, 100 * cmb.WinRatio.values, yerr=100*cmb['2se'], fmt='k-', marker='o', mec='k', mfc='w', ecolor='gray', elinewidth=0.5, capsize=2)
plt.xlabel('Age')
plt.ylabel('Win Percentage')
plt.xlim(xmin, xmax)
minor_ticks = np.arange(xmin, xmax, 1)
ax.set_xticks(minor_ticks, minor = True)
major_ticks = np.arange(xmin, xmax+2, 2)
ax.set_xticks(major_ticks)
ax.set_xticklabels(major_ticks)
plt.savefig('report/age/win_percent_vs_age.pdf', bbox_inches='tight')

win_count_by_age = wins.Age_int.value_counts()

# count fights per age without double counting
ages = win_count_by_age.index
counts = []
for age in ages:
     ct = wins[(wins.Age_int == age) | (wins.Age_L_int == age)].shape[0]
     counts.append(ct)
# total_count_by_age = pd.Series(data=counts, index=ages)
# win percentage is number of wins by age o
total_count_by_age = win_count_by_age + wins.Age_L_int.value_counts()
win_percent_by_age = win_count_by_age / total_count_by_age
cmb = pd.concat([win_count_by_age, total_count_by_age, win_percent_by_age], axis=1).sort_index()
cmb = cmb.loc[20:40]
cmb.columns = ['wins', 'total', 'WinRatio']
cmb['2se'] = -t.ppf(0.025, cmb.total - 1) * (cmb.WinRatio * (1.0 - cmb.WinRatio) / cmb.total)**0.5
cmb

cmb['losses'] = cmb.total - cmb.wins
cont_table = cmb[['wins', 'losses']].T
cont_table

chi_sq, p_value, dof, expect = chi2_contingency(cont_table)
print chi_sq, p_value, p_value > 0.05

x = cmb.index
y = 100 * cmb.WinRatio.values
xmin = 19
xmax = 41

m, b = np.polyfit(x, y, 1)
fig, ax = plt.subplots()
plt.plot([xmin, xmax], [50, 50], 'k:')
plt.plot(np.linspace(xmin, xmax), m * np.linspace(xmin, xmax) + b, 'k-')
plt.errorbar(cmb.index, 100 * cmb.WinRatio.values, yerr=100*cmb['2se'], fmt='o', marker='o', mec='k', mfc='w', ecolor='gray', elinewidth=0.5, capsize=2)
#plt.plot(cmb.index, 100 * cmb.WinRatio.values, 'wo', mec='k')
plt.xlabel('Age')
plt.ylabel('Win percentage')
plt.xlim(xmin, xmax)
plt.ylim(0, 100)
minor_ticks = np.arange(xmin, xmax, 1)
ax.set_xticks(minor_ticks, minor = True)
major_ticks = np.arange(20, 42, 2)
ax.set_xticks(major_ticks)
ax.set_xticklabels(major_ticks)
plt.savefig('report/age/win_percent_vs_age2.pdf', bbox_inches='tight')

float(wins[wins.Age_int == 40].shape[0]) / (wins[wins.Age_int == 40].shape[0] + wins[wins.Age_L_int == 40].shape[0])

wins[wins.Age_int == 40].shape[0] + wins[wins.Age_L_int == 40].shape[0]

from scipy.stats import pearsonr, spearmanr

corr_pearson, p_value_pearson = pearsonr(x, y)
corr_spearman, p_value_spearman = spearmanr(x, y)
print corr_pearson, p_value_pearson
print corr_spearman, p_value_spearman

w = win_count_by_age[total_count_by_age > 20].sort_index()
tot = total_count_by_age[total_count_by_age > 20].sort_index()
cont_table = pd.DataFrame({'wins':w, 'total':tot}).T.sort_index(ascending=False)
cont_table

chi_sq, p_value, dof, expect = chi2_contingency(cont_table)
print chi_sq, p_value, p_value > 0.05

N = cont_table.sum().sum()
V = (chi_sq / (N * min(2 - 1, 21 - 1)))**0.5
V

def two_sided_binom(x):
     wins = x[0]
     total = x[1]
     if wins / total == 0.5:
          return 1.0
     elif wins / total < 0.5:
          return 2 * binom.cdf(p=0.5, k=wins, n=total)
     else:
          return 2 * (1.0 - binom.cdf(p=0.5, k=wins-1, n=total))

cont_table.loc['p_value'] = cont_table.apply(two_sided_binom, axis=0)
cont_table.applymap(lambda x: round(x, 2))

flips = 21
k_values = range(flips + 1)
plt.vlines(x, ymin=0, ymax=[binom.pmf(k, p=0.5, n=flips) for k in k_values], lw=4)
plt.xlabel('k')
plt.ylabel('P(k)')

wins['AgeDiff'] = wins.Age_L - wins.Age
wins.AgeDiff = wins.AgeDiff.apply(round)
delta_age = wins.AgeDiff.value_counts().sort_index()
delta_age

delta_age_overall = np.abs(wins.AgeDiff).value_counts().sort_index()
delta_age_overall

younger_diff = delta_age.loc[0:17]
younger_diff

cnt = pd.concat([younger_diff, delta_age_overall, younger_diff / delta_age_overall], axis=1).sort_index()
cnt = cnt.loc[1:12]
cnt.columns = ['younger_wins', 'total', 'WinRatio']
cnt['younger_losses'] = cnt.total - cnt.younger_wins
cnt['2se'] = -t.ppf(0.025, cnt.total - 1) * (cnt.WinRatio * (1.0 - cnt.WinRatio) / cnt.total)**0.5
cnt

xmin = 0
xmax = 13

m, b = np.polyfit(cnt.index, 100 * cnt.WinRatio, 1)
fig, ax = plt.subplots()
plt.plot([xmin, xmax], [50, 50], 'k:')
plt.plot(np.linspace(xmin, xmax), m * np.linspace(xmin, xmax) + b, 'k-')
plt.errorbar(cnt.index, 100 * cnt.WinRatio.values, fmt='o', color='k', marker='o', mfc='w', yerr=100*cnt['2se'], ecolor='gray', elinewidth=0.5, capsize=2)
plt.plot(cnt.index, 100 * cnt.WinRatio, 'wo')
plt.xlim(xmin, xmax)
plt.xlim(0, 13)
plt.ylim(20,80)
plt.xticks(range(1, 13))
plt.xlabel('Age Difference (years)')
plt.ylabel('Win Percentage\nof Younger Fighter')
plt.savefig('report/age/win_percent_of_younger.pdf', bbox_inches='tight')

wins[wins.AgeDiff == 4][['Winner', 'Loser', 'Age', 'Age_L', 'AgeDiff']].shape[0]

wins[wins.AgeDiff == -4][['Winner', 'Loser', 'Age', 'Age_L', 'AgeDiff']].shape[0]

221 / (221 + 169.0)

corr_pearson, p_value_pearson = pearsonr(x, y)
corr_spearman, p_value_spearman = spearmanr(x, y)
print corr_pearson, p_value_pearson
print corr_spearman, p_value_spearman

binom.cdf(p=0.5, k=257, n=257+287)

cont_table = cnt[['younger_wins', 'younger_losses']].copy()
cont_table

chi_sq, p_value, dof, expect = chi2_contingency(cont_table)
print chi_sq, p_value, p_value > 0.05

N = cont_table.sum().sum()
V = (chi_sq / (N * min(2 - 1, 14 - 1)))**0.5
V

def two_sided_binom(x):
     wins = float(x[0])
     total = x[0] + x[1]
     if wins / total == 0.5:
          return 1.0
     elif wins / total < 0.5:
          return 2 * binom.cdf(p=0.5, k=wins, n=total)
     else:
          return 2 * (1.0 - binom.cdf(p=0.5, k=wins-1, n=total))

cont_table['p_value'] = cont_table.apply(two_sided_binom, axis=1)
cont_table.applymap(lambda x: round(x, 2))

fights[fights.Winner == 'Robbie Lawler']

all_wins = df[pd.notnull(df.Dob)].copy()
all_wins['Age'] = (all_wins.Date - all_wins.Dob) / np.timedelta64(1, 'Y')

all_loses = df[pd.notnull(df.Dob_L)].copy()
all_loses['Age'] = (all_loses.Date - all_loses.Dob_L) / np.timedelta64(1, 'Y')

youngest_winners = all_wins.groupby('Winner').agg({'Age':min})
youngest_losers = all_loses.groupby('Loser').agg({'Age':min})
youngest = youngest_winners.append(youngest_losers).reset_index()
youngest = youngest.groupby('index').agg({'Age':min}).sort_values('Age').applymap(lambda x: round(x, 1)).reset_index()[:30]
youngest

oldest_winners = all_wins.groupby('Winner').agg({'Age':max})
oldest_losers = all_loses.groupby('Loser').agg({'Age':max})
oldest = oldest_winners.append(oldest_losers).reset_index()
oldest = oldest.groupby('index').agg({'Age':max}).sort_values('Age', ascending=False).applymap(lambda x: round(x, 1)).reset_index()[:30]
oldest

young_old = youngest.merge(oldest, left_index=True, right_index=True)
young_old.index = range(1, 31)
young_old.columns = ['Youngest', 'Age', 'Oldest', 'Age']
young_old.to_latex('report/age/youngest_oldest_RAW.tex')
young_old

