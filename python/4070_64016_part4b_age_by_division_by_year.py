import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

iofile = 'data/fightmetric_cards/fightmetric_fights_CLEAN_3-6-2017.csv'
fights = pd.read_csv(iofile, header=0, parse_dates=['Date'])
fights.head(3)

iofile = 'data/fightmetric_fighters_with_corrections_from_UFC_Wikipedia_CLEAN.csv'
fighters = pd.read_csv(iofile, header=0, parse_dates=['Dob'])
fighters.head(3)

cols = ['Name', 'Weight', 'Height', 'Reach', 'LegReach', 'Stance', 'Dob']
df = fights.merge(fighters[cols], how='left', left_on='Winner', right_on='Name')
df.shape

df = df.merge(fighters[cols], how='left', left_on='Loser', right_on='Name', suffixes=('', '_L'))
df.shape

df = df.drop(['Name', 'Name_L'], axis=1)

df['Age'] = (df.Date - df.Dob) / np.timedelta64(1, 'Y')
df['Age_L'] = (df.Date - df.Dob_L) / np.timedelta64(1, 'Y')

wc = ["Women's Strawweight", "Women's Bantamweight", 'Flyweight', 'Bantamweight', 'Featherweight',
      'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight']
years = range(1993, 2017)
for w in wc:
     w_class = w.lower().replace('women\'s', 'w').replace(' ', '_')
     exec('num_fights_' + w_class + ' = []')
     exec('age_' + w_class + ' = []')
     exec('height_' + w_class + ' = []')
     exec('reach_' + w_class + ' = []')
     for year in years:
          recs = df[(df.WeightClass == w) & (df.Date.dt.year == year)]
          exec('num_fights_' + w_class + '.append(recs.shape[0])')
          exec('age_' + w_class + '.append(recs.Age.append(recs.Age_L).mean())')
          exec('height_' + w_class + '.append(recs.Height.append(recs.Height_L).mean())')
          exec('reach_' + w_class + '.append(recs.Reach.append(recs.Reach_L).mean())')
     exec('num_fights_' + w_class + '= np.array(num_fights_' + w_class + ', dtype=np.float)')
     exec('num_fights_' + w_class + '[num_fights_' + w_class + '==0] = np.nan')

for w in ['Featherweight', 'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight']:
     w_class = w.lower().replace('women\'s', 'w').replace(' ', '_')

     fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(5, 7), sharex='all')
     
     exec('ax4.plot(years, num_fights_'+ w_class +', \'k-\',  marker=\'o\', mec=\'k\', mfc=\'w\',mew=1, ms=7)')
     exec('ax3.plot(years, age_'+ w_class +', \'r-\', marker=\'o\', mec=\'r\', mfc=\'w\', mew=1, ms=7)')
     exec('ax2.plot(years, height_'+ w_class +', \'g-\', marker=\'o\', mec=\'g\', mfc=\'w\', mew=1, ms=7)')
     exec('ax1.plot(years, reach_'+ w_class +', \'b-\', marker=\'o\', mec=\'b\', mfc=\'w\', mew=1, ms=7)')

     ax4.set_ylabel('Fights')
     ax3.set_ylabel('Age')
     ax2.set_ylabel('Height')
     ax1.set_ylabel('Reach')
     ax4.set_xlabel('Year')
     
     if w == 'Featherweight':
          ax1.set_ylim(69, 71)
          major_ticks_ = np.arange(69.5, 72, 0.5)
          ax1.set_yticks(major_ticks_)
          ax1.set_yticklabels(major_ticks_)
          
          ax2.set_ylim(67.5, 69.5)
          major_ticks_ = np.arange(68, 70, 0.5)
          ax2.set_yticks(major_ticks_)
          ax2.set_yticklabels(major_ticks_)

     if w == 'Lightweight':
          ax1.set_ylim(69, 73)
          major_ticks_ = np.arange(70, 74, 1)
          ax1.set_yticks(major_ticks_)
          ax1.set_yticklabels(major_ticks_)
          
          ax2.set_ylim(66, 72)
          major_ticks_ = np.arange(67, 72, 1)
          ax2.set_yticks(major_ticks_)
          ax2.set_yticklabels(major_ticks_)

     if w == 'Welterweight':
          ax1.set_ylim(69, 74)
          major_ticks_ = np.arange(70, 75, 1)
          ax1.set_yticks(major_ticks_)
          ax1.set_yticklabels(major_ticks_)
          
          ax2.set_ylim(68, 73)
          major_ticks_ = np.arange(69, 73, 1)
          ax2.set_yticks(major_ticks_)
          ax2.set_yticklabels(major_ticks_)

     if w == 'Middleweight':
          ax1.set_ylim(73.5, 75.5)
          major_ticks_ = np.arange(74, 76, 0.5)
          ax1.set_yticks(major_ticks_)
          ax1.set_yticklabels(major_ticks_)
          
          ax2.set_ylim(70.5, 73.5)
          major_ticks_ = np.arange(71, 74, 1)
          ax2.set_yticks(major_ticks_)
          ax2.set_yticklabels(major_ticks_)

     if w == 'Light Heavyweight':
          ax1.set_ylim(74, 76.5)
          major_ticks_ = np.arange(74.5, 77, 0.5)
          ax1.set_yticks(major_ticks_)
          ax1.set_yticklabels(major_ticks_)
          
          ax2.set_ylim(72, 74.5)
          major_ticks_ = np.arange(72.5, 74.5, 0.5)
          ax2.set_yticks(major_ticks_)
          ax2.set_yticklabels(major_ticks_)
          
     if w == 'Heavyweight':
          ax1.set_ylim(73.5, 79)
          major_ticks_ = np.arange(74, 80, 1)
          ax1.set_yticks(major_ticks_)
          ax1.set_yticklabels(major_ticks_)
          
          ax2.set_ylim(72, 77)
          major_ticks_ = np.arange(73, 77, 1)
          ax2.set_yticks(major_ticks_)
          ax2.set_yticklabels(major_ticks_)

     fig.subplots_adjust(hspace=0)
     ax1.set_title(w, fontsize=14)
     ax3.set_ylim(24, 34)
     major_ticks_ = np.arange(26, 34, 2)
     ax3.set_yticks(major_ticks_)
     ax3.set_yticklabels(major_ticks_)
     
     ax4.set_xlim(1996, 2020)
     major_ticks = np.arange(1996, 2020, 4)
     ax4.set_xticks(major_ticks)
     minor_ticks = np.arange(1996, 2020, 1)
     ax4.set_xticks(minor_ticks, minor = True)
     ax4.set_ylim(0, 125)
     major_ticks_ = np.arange(25, 125, 25)
     ax4.set_yticks(major_ticks_)
     ax4.set_yticklabels(major_ticks_)
     
     #ax1.margins(0.1)
     #ax2.margins(0.1)
     #ax4.margins(0.1)
     
     plt.savefig('report/age/' + w_class + '_age_height_reach.pdf', bbox_inches='tight')

small = ['Flyweight', 'Bantamweight', "Women's Strawweight", "Women's Bantamweight"]
for w in small:
     w_class = w.lower().replace('women\'s', 'w').replace(' ', '_')

     fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(3, 7), sharex='all')
     
     exec('ax1.plot(years, num_fights_'+ w_class +', \'k-\',  marker=\'o\', mec=\'k\', mfc=\'w\',mew=1, ms=7)')
     exec('ax2.plot(years, age_'+ w_class +', \'r-\', marker=\'o\', mec=\'r\', mfc=\'w\', mew=1, ms=7)')
     exec('ax3.plot(years, height_'+ w_class +', \'g-\', marker=\'o\', mec=\'g\', mfc=\'w\', mew=1, ms=7)')
     exec('ax4.plot(years, reach_'+ w_class +', \'b-\', marker=\'o\', mec=\'b\', mfc=\'w\', mew=1, ms=7)')

     ax2.set_ylim(24, 34)

     ax1.set_ylabel('Fights')
     ax2.set_ylabel('Age')
     ax3.set_ylabel('Height')
     ax4.set_ylabel('Reach')
     ax4.set_xlabel('Year')

     plt.setp(ax1.get_yticklabels()[0], visible=False)
     plt.setp(ax2.get_yticklabels()[0], visible=False)
     plt.setp(ax3.get_yticklabels()[0], visible=False)
     plt.setp(ax4.get_yticklabels()[0], visible=False)

     plt.setp(ax1.get_yticklabels()[-1], visible=False)
     plt.setp(ax2.get_yticklabels()[-1], visible=False)
     plt.setp(ax3.get_yticklabels()[-1], visible=False)
     plt.setp(ax4.get_yticklabels()[-1], visible=False)

     fig.subplots_adjust(hspace=0)
     ax1.set_title(w, fontsize=14)
     #ax4.set_xlim(2010, 2018)
     #major_ticks = np.arange(2010, 2018, 2)
     #ax4.set_xticks(major_ticks)
     #minor_ticks = np.arange(1996, 2020, 1)
     #ax4.set_xticks(minor_ticks, minor = True)
     plt.savefig('report/age/' + w_class + '_age_height_reach.pdf', bbox_inches='tight')

name_weight = pd.read_csv('data/weight_class_majority.csv', header=0)
name_weight[name_weight.Active == 1].shape[0]

fighters = fighters.merge(name_weight, on='Name', how='right')
fighters['Age'] = (pd.to_datetime('today') - fighters.Dob) / np.timedelta64(1, 'Y')

age_class = []
for w in wc:
     tmp = fighters[(fighters.WeightClassMajority == w) & (fighters.Active == 1)]
     w_class = w.lower().replace('women\'s', 'w').replace(' ', '_')
     exec('age_' + w_class + '=tmp.Age.values')
     exec('age_class.append(age_' + w_class + ')')

mean_age = fighters[(fighters.Active == 1)].Age.mean()
fig, ax = plt.subplots(figsize=(8, 4))
wlabels = ['W-SW', 'W-BW', 'FYW', 'BW', 'FTW', 'LW', 'WW', 'MW', 'LH', 'HW']
plt.boxplot(age_class, labels=wlabels, patch_artist=True)
plt.plot([-1, 13], [mean_age, mean_age], 'k:', zorder=0)
for i, ages in enumerate(age_class):
     plt.text(i + 1, 16, ages.size, ha='center', fontsize=10)
plt.ylim(15, 45)
plt.xlabel('Weight Class')
plt.ylabel('Age (years)')
plt.savefig('report/age/anova_age_by_weightclass.pdf', bbox_inches='tight')

from scipy.stats import levene, bartlett

W, p_value = levene(*age_class, center='mean')
W, p_value, p_value > 0.05

W, p_value = bartlett(*age_class)
W, p_value, p_value > 0.05

from scipy.stats import kurtosis, skew, kurtosistest

for ac in age_class:
     Z, p_value = kurtosistest(ac)
     print '%.1f\t%.1f\t%.1f\t%.1f\t%.1f' % (ac.mean(), ac.std(), skew(ac), kurtosis(ac), p_value)

from scipy.stats import f_oneway

F_statistic, p_value = f_oneway(*age_class)
F_statistic, p_value, p_value > 0.05

with open('data/ranked_ufc_fighters_1488838405.txt') as f:
     ranked = f.readlines()
ranked = [fighter.strip() for fighter in ranked]

wc = ['Flyweight', 'Bantamweight', 'Featherweight','Lightweight', 'Welterweight', 'Middleweight',
      'Light Heavyweight', 'Heavyweight', "Women's Strawweight", "Women's Bantamweight"]
for i, w in enumerate(wc):
     w_class = w.lower().replace('women\'s', 'w').replace(' ', '_')
     exec('ranked_' + w_class + '=[]')
     for j in range(16):
          exec('ranked_' + w_class + '.append(ranked[i * 16 + j])')

for w in wc:
     w_class = w.lower().replace('women\'s', 'w').replace(' ', '_')
     exec('x=fighters[fighters.Name.isin(ranked_' + w_class + ')].Reach.mean()')
     exec("y=fighters[(fighters.WeightClassMajority ==\""  + w + "\") & (fighters.Active == 1) & (~fighters.Name.isin(ranked_" + w_class + "))].Reach.mean()")
     print w_class, x, y, x > y

