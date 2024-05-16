import numpy as np
import pandas as pd
df = pd.read_csv('nc.csv')

df.head()

df.count()

df.describe()

w = df.weight
nonsmoker = df.habit == 'nonsmoker'
smoker = df.habit == 'smoker'

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['font.size'] = 14

n, bins, patches = plt.hist((w[smoker], w[nonsmoker]), label=('smoker', 'nonsmoker'), bins=15, normed=True)
plt.xlabel('Weight')
plt.ylabel('Count')
plt.legend(loc='upper left')

smoke_mean = w[smoker].mean()
smoke_std = w[smoker].std()
smoke_n = w[smoker].size
nonsmoke_mean = w[nonsmoker].mean()
nonsmoke_std = w[nonsmoker].std()
nonsmoke_n = w[nonsmoker].size
print smoke_mean, nonsmoke_mean
print smoke_std, nonsmoke_std
print smoke_n, nonsmoke_n

SE = np.sqrt(smoke_std**2 / smoke_n + nonsmoke_std**2 / nonsmoke_n)
print SE

T = ((nonsmoke_mean - smoke_mean) - 0.0) / SE
print T

from scipy.stats import t
p_value = 2 * (1.0 - t.cdf(T, min(smoke_n, nonsmoke_n) - 1))
print p_value, p_value > 0.05

import scipy.stats
t_stat, p_value = scipy.stats.ttest_ind(w[smoker], w[nonsmoker], equal_var=False)
print t_stat, p_value

