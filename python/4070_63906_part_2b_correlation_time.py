import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

from helper_methods import drop_duplicate_inspections

df = pd.read_csv('data/training_labels.txt', parse_dates=['date'])
df = df.sort_values(['restaurant_id', 'date'])
df = drop_duplicate_inspections(df, threshold=60)
df = df[(df.date >= pd.to_datetime('2008-01-01')) & (df.date <= pd.to_datetime('2014-12-31'))]
df.head()

df.info()

plt.plot(df[df.restaurant_id == '1JEbamOR'].date, df[df.restaurant_id == '1JEbamOR']['*'], 'k:', label='*', marker='o')
plt.plot(df[df.restaurant_id == '1JEbamOR'].date, df[df.restaurant_id == '1JEbamOR']['**'], 'r:', label='**', marker='o')
plt.plot(df[df.restaurant_id == '1JEbamOR'].date, df[df.restaurant_id == '1JEbamOR']['***'], 'g:', label='***', marker='o')
plt.xlabel('Date')
plt.ylabel('Number of violations')
plt.legend(loc='upper left')
plt.title('1JEbamOR')

plt.plot(df[df.restaurant_id == '0ZED0WED'].date, df[df.restaurant_id == '0ZED0WED']['*'], 'k:', label='*', marker='o')
plt.plot(df[df.restaurant_id == '0ZED0WED'].date, df[df.restaurant_id == '0ZED0WED']['**'], 'r:', label='**', marker='o')
plt.plot(df[df.restaurant_id == '0ZED0WED'].date, df[df.restaurant_id == '0ZED0WED']['***'], 'g:', label='***', marker='o')
plt.xlabel('Date')
plt.ylabel('Number of violations')
plt.legend(loc='upper left')
plt.title('0ZED0WED')

# https://www.wunderground.com/history/airport/KBOS/2015/1/1/CustomHistory.html
bos_wthr = pd.read_csv('data/boston_weather_2015_2011.csv', parse_dates=['EST'])
bos_wthr['weekofyear'] = bos_wthr['EST'].apply(lambda x: x.weekofyear)
bos_wthr.head(3).transpose()

df['weekofyear'] = df.date.apply(lambda x: x.weekofyear)
weekofyear_violations = df.groupby('weekofyear').agg({'*':[np.size, np.mean], '**':[np.mean], '***':[np.mean]})
weekofyear_violations.head()

fig, ax1 = plt.subplots()
ax1.bar(weekofyear_violations.index, weekofyear_violations[('*', 'size')], width=1)
ax1.set_xlabel('Week of the year')
ax1.set_ylabel('Number of inspections', color='b')

mean_T_by_week = bos_wthr.groupby('weekofyear').agg({'Mean TemperatureF': [np.mean]})
ax2 = ax1.twinx()
ax2.plot(bos_wthr.EST.apply(lambda x: x.dayofyear / 7.0), bos_wthr['Mean TemperatureF'], 'm.', alpha=0.25, ms=5)
ax2.plot(mean_T_by_week.index, mean_T_by_week, 'r', marker='o')
ax2.set_ylabel('Mean temperature 2011-2015 (F)', color='r')
ax2.set_xlim(0, 55)
ax2.set_ylim(0, 100)

from scipy.stats import pearsonr, spearmanr
print pearsonr(mean_T_by_week[('Mean TemperatureF', 'mean')][1:-2], weekofyear_violations[('*', 'size')][1:-2])
print spearmanr(mean_T_by_week[('Mean TemperatureF', 'mean')], weekofyear_violations[('*', 'size')])

colors = ['b', 'g', 'r']
stars = ['*', '**', '***']
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
for i in range(3):
    ax[i].bar(weekofyear_violations.index, weekofyear_violations[(stars[i], 'mean')], width=1, color=colors[i])
    ax[i].set_xlabel('Week of the year')
    ax[i].set_ylabel('Number of violations (' + stars[i] + ')')
    ax[i].set_xlim(0, 53)
plt.tight_layout()

for star in stars:
    print star, pearsonr(mean_T_by_week[('Mean TemperatureF', 'mean')][1:-2], weekofyear_violations[(star, 'mean')][1:-2])
    print star, spearmanr(mean_T_by_week[('Mean TemperatureF', 'mean')], weekofyear_violations[(star, 'mean')])

d = df[df.restaurant_id == '0ZED0WED']
d = d.sort_values('date')
d.reset_index(inplace=True, drop=True)
d

for index, row in d.iterrows():
    print index, row['*']

from collections import defaultdict
ct = defaultdict(int)
cf = defaultdict(int)
cf_list = defaultdict(list)

for rest_id in df.restaurant_id.unique():
    d = df[df.restaurant_id == rest_id]
    d.sort_values('date')
    d.reset_index(inplace=True, drop=True)
    num_inspect = d.shape[0]
    for i in xrange(num_inspect - 1):
        mean_one_star = d.iloc[i:].mean()['*']
        one_star = d.ix[i, '*'] - mean_one_star
        t_start = d.ix[i, 'date']
        for j in xrange(i, num_inspect):
            t_diff = int((d.ix[j, 'date'] - t_start) / np.timedelta64(1, 'W'))
            cf[t_diff] += (d.ix[j, '*'] - mean_one_star) * one_star
            ct[t_diff] += 1
            cf_list[t_diff].append((d.ix[j, '*'] - mean_one_star) * one_star)

# compute error bars
std_err = [np.sqrt(np.var(np.array(cf_list[t]) / (cf[0] / ct[0])) / ct[t]) for t in sorted(ct.keys())]

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex='all')
ax1.errorbar(sorted(ct.keys()), [cf[t] / ct[t] / (cf[0] / ct[0]) for t in sorted(ct.keys())], marker='o', yerr=std_err, ecolor='k')
ax1.plot([0, 25], [0, 0], 'k:')
ax1.set_ylabel(r'$\frac{\langle(x(t) - \bar{x})(x(0) - \bar{x})\rangle}{\langle (x(0) - \bar{x})(x(0) - \bar{x})\rangle}$', fontsize=32)
ax1.set_xlim(0, 25)
ax1.set_ylim(-1, 1.5)

ax2.bar(np.array(ct.keys()) - 0.5, ct.values(), width=1, )
ax2.set_xlim(-0.5, 25.5)
ax2.set_ylim(0, 1200)
ax2.set_xlabel('Weeks between inspections')
ax2.set_ylabel('Count')

fig.subplots_adjust(hspace=0)
plt.setp(ax1.get_yticklabels()[0], visible=False)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 5))
plt.bar(ct.keys(), ct.values(), width=1)
plt.xlabel('Weeks between inspections')
plt.ylabel('Count')
plt.ylim(0, 1200)

auc = [0.5]
running_sum = 0.5
for t in sorted(ct.keys()[1:]):
    running_sum += cf[t] / ct[t] / (cf[0] / ct[0])
    auc.append(running_sum)
plt.plot(sorted(ct.keys()), auc)
plt.plot([0, 25], [0, 0], 'k:')
plt.xlabel('Weeks between inspections')
plt.ylabel(r'$\int_0^{\infty} c(t)dt$', fontsize=24)
plt.xlim(0, 25)
plt.ylim(-1, 5)

