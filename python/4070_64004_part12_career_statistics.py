import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)

fights[(fights.Winner == 'Anderson Silva') | (fights.Loser == 'Anderson Silva')].shape[0]

df = pd.read_csv('data/fightmetric_career_stats.csv', header=0)
df.head(10)

fs = fights[fights.Date > pd.to_datetime('2005-01-01')]
win_lose = fs.Winner.append(fs.Loser)
num_fights = win_lose.value_counts().to_frame()
num_fights.columns = ['total_fights']
num_fights.size

# Dorian Price is the only UFC fighter after 2005 with a total of 0 because he lost in 23 seconds
df['total'] = df.sum(axis=1)

cs = num_fights.merge(df, left_index=True, right_on='Name', how='left').reset_index(drop=True)
cs.head()

cs.shape

cs.info()

cs5 = cs[cs.total_fights > 5]

st = cs5.sort_values('slpm', ascending=False).head(15)
st = st[['Name', 'slpm', 'total_fights']]
st.columns = ['Name', 'Strikes Landed per Minute', 'Total Fights']
st.index = range(1, 16)
st.to_latex('report/offense_defense/most_slpm_RAW.tex')

st = cs5.sort_values('str_acc', ascending=False).head(15)
st = st[['Name', 'str_acc', 'total_fights']]
st.str_acc = st.str_acc * 100
st.str_acc = st.str_acc.astype(int)
st.columns = ['Name', 'Striking Accuracy (%)', 'Total Fights']
st.index = range(1, 16)
st.to_latex('report/offense_defense/most_acc_str_RAW.tex')

st = cs5.sort_values('td_avg', ascending=False).head(15)
st = st[['Name', 'td_avg', 'total_fights']]
st.columns = ['Name', 'Takedowns per 15 Minutes', 'Total Fights']
st.index = range(1, 16)
st.to_latex('report/offense_defense/most_td_RAW.tex')

st = cs5.sort_values('sub_avg', ascending=False).head(15)
st = st[['Name', 'sub_avg', 'total_fights']]
st.columns = ['Name', 'Submission Attempts per 15 Minutes', 'Total Fights']
st.index = range(1, 16)
st.to_latex('report/offense_defense/most_subs_RAW.tex')

plt.close('all')
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2, figsize=(9, 12))

ax1.hist(cs5.slpm, bins=np.linspace(0, 8, num=20), rwidth=0.8)
ax1.set_xlabel('Strikes Landed Per Minute')
ax1.set_ylabel('Count')

ax2.hist(cs5.sapm, bins=np.linspace(0, 8, num=20), rwidth=0.8)
ax2.set_xlabel('Strikes Absorbed Per Minute')
ax2.set_ylabel('Count')

ax3.hist(100*cs5.str_acc, bins=np.linspace(0, 100, num=20), rwidth=0.8)
ax3.set_xlabel('Striking Accuracy (%)')
ax3.set_ylabel('Count')

ax4.hist(100*cs5.str_def, bins=np.linspace(0, 100, num=20), rwidth=0.8)
ax4.set_xlabel('Striking Defense (%)')
ax4.set_ylabel('Count')

ax5.hist(cs5.td_avg, bins=np.linspace(0, 8, num=20), rwidth=0.8)
ax5.set_xlabel('Takedowns per 15 Minutes')
ax5.set_ylabel('Count')

ax6.hist(100*cs5.td_acc, bins=np.linspace(0, 100, num=20), rwidth=0.8)
ax6.set_xlabel('Takedown Accuracy (%)')
ax6.set_ylabel('Count')

ax7.hist(100*cs5.td_def, bins=np.linspace(0, 100, num=20), rwidth=0.8)
ax7.set_xlabel('Takedowns Defense (%)')
ax7.set_ylabel('Count')

ax8.hist(cs5.sub_avg, bins=np.linspace(0, 6, num=20), rwidth=0.8)
ax8.set_xlabel('Submission Attempts per 15 Minutes')
ax8.set_ylabel('Count')

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)

plt.savefig('report/offense_defense/many_hist.pdf', bbox_inches='tight')

cs5.slpm.quantile(0.5)

cs5.sort_values('str_acc', ascending=False).head(15)

cs5.sort_values('sapm', ascending=False).head(15)

plt.hist(cs5.sapm, bins=np.linspace(0, 8, num=20), rwidth=0.8)
plt.xlabel('Strikes Absorbed Per Minute')
plt.ylabel('Count')

cs5.sapm.quantile(0.5)

cs5.sort_values('str_def', ascending=False).head(15)

cs5.sort_values('td_avg', ascending=False).head(15)

plt.hist(cs5.td_avg, bins=np.linspace(0, 8, num=20), rwidth=0.8)
plt.xlabel('Average Number of Takedowns\nLanded per 15 Minutes')
plt.ylabel('Count')

cs5.td_avg.quantile(0.5)

from scipy import stats
stats.percentileofscore(cs5.td_avg, 1.55)

stats.percentileofscore(cs5.td_avg, 5.0)

cs5.sort_values('td_acc', ascending=False).head(15)

cs5.sort_values('td_def', ascending=False).head(15)

cs5.sort_values('sub_avg', ascending=False).head(15)

plt.hist(cs5.sub_avg, bins=np.linspace(0, 6, num=20), rwidth=0.8)
plt.xlabel('Average Submissions Attempted per 15 Minutes')
plt.ylabel('Count')

cs5.sub_avg.quantile(0.5)

mj = pd.read_csv('data/weight_class_majority.csv', header=0)
mj.head()

cmb = cs5.merge(mj, on='Name', how='left')
cmb.head()

cmb.isnull().sum()

cmb.groupby('WeightClassMajority').count()

by_weight = cmb.groupby('WeightClassMajority').median()
by_weight = by_weight.drop("Women's Strawweight")
by_weight

wc = ["Women's Strawweight", "Women's Bantamweight", 'Flyweight', 'Bantamweight', 'Featherweight',
      'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight']
wlabels = ['W-S', 'W-B', 'FY', 'BW', 'FTW', 'LW', 'WW', 'MW', 'LH', 'HW']

wlabels

wc.reverse()
wlabels.reverse()
by_weight = by_weight.reindex(wc)
by_weight

wc

wlabels

for i in range(9):
     #plt.bar(range(by_weight.shape[0]), by_weight.slpm)
     plt.bar(i, by_weight.sub_avg.iloc[i])
plt.axes().set_xticks(range(by_weight.shape[0] - 1))
plt.axes().set_xticklabels(wlabels[:-1])
plt.xlabel('Weight Class')
plt.ylabel('Submissions per 15 Minutes')

plt.close('all')
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2, figsize=(9, 12))

for i in range(9):
     ax1.bar(i, by_weight.slpm.iloc[i], width=0.6)
ax1.set_xticks(range(by_weight.shape[0] - 1))
ax1.set_xticklabels(wlabels[:-1])
ax1.set_ylabel('Strikes Landed per Minute')
ax1.set_ylim(0, 4)

for i in range(9):
     ax2.bar(i, by_weight.sapm.iloc[i], width=0.6)
ax2.set_xticks(range(by_weight.shape[0] - 1))
ax2.set_xticklabels(wlabels[:-1])
ax2.set_ylabel('Strikes Absorbed per Minute')
ax2.set_ylim(0, 4)

for i in range(9):
     ax3.bar(i, 100*by_weight.str_acc.iloc[i], width=0.6)
ax3.set_xticks(range(by_weight.shape[0] - 1))
ax3.set_xticklabels(wlabels[:-1])
ax3.set_ylabel('Striking Accuracy (%)')
ax3.set_ylim(0, 70)

for i in range(9):
     ax4.bar(i, 100*by_weight.str_def.iloc[i], width=0.6)
ax4.set_xticks(range(by_weight.shape[0] - 1))
ax4.set_xticklabels(wlabels[:-1])
ax4.set_ylabel('Striking Defense (%)')
ax4.set_ylim(0, 70)

for i in range(9):
     ax5.bar(i, by_weight.td_avg.iloc[i], width=0.6)
ax5.set_xticks(range(by_weight.shape[0] - 1))
ax5.set_xticklabels(wlabels[:-1])
ax5.set_ylabel('Takedowns per 15 Minutes')

for i in range(9):
     ax6.bar(i, 100*by_weight.td_acc.iloc[i], width=0.6)
ax6.set_xticks(range(by_weight.shape[0] - 1))
ax6.set_xticklabels(wlabels[:-1])
ax6.set_ylabel('Takedown Accuracy (%)')

for i in range(9):
     ax7.bar(i, 100*by_weight.td_def.iloc[i], width=0.6)
ax7.set_xticks(range(by_weight.shape[0] - 1))
ax7.set_xticklabels(wlabels[:-1])
ax7.set_ylabel('Takedown Defense (%)')

for i in range(9):
     ax8.bar(i, by_weight.sub_avg.iloc[i], width=0.6)
ax8.set_xticks(range(by_weight.shape[0] - 1))
ax8.set_xticklabels(wlabels[:-1])
ax8.set_ylabel('Submissions per 15 Minutes')

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)

plt.savefig('report/offense_defense/many_bars.pdf', bbox_inches='tight')

age_class = []
for w in wc:
     tmp = cmb[(cmb.WeightClassMajority == w) & (cmb.Active == 1)]
     w_class = w.lower().replace('women\'s', 'w').replace(' ', '_')
     exec('age_' + w_class + '=tmp.td_avg.values')
     exec('age_class.append(age_' + w_class + ')')

mean_age = cmb[(cmb.Active == 1)].td_avg.mean()
fig, ax = plt.subplots(figsize=(8, 4))
wlabels = ['W-SW', 'W-BW', 'FYW', 'BW', 'FTW', 'LW', 'WW', 'MW', 'LH', 'HW']
plt.boxplot(age_class, labels=wlabels, patch_artist=True)
plt.plot([-1, 13], [mean_age, mean_age], 'k:', zorder=0)
for i, ages in enumerate(age_class):
     plt.text(i + 1, 6.7, ages.size, ha='center', fontsize=10)
#plt.ylim(15, 45)
plt.xlabel('Weight Class')
plt.ylabel('Age (years)')
#plt.savefig('report/finish/anova_age_by_weightclass.pdf', bbox_inches='tight')

