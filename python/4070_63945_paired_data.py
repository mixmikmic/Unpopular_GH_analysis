import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

df = pd.read_csv('textbooks.txt', sep='\t')
df.head()

df.describe()

plt.plot([0, 250], [0, 250], 'k:')
plt.plot(df.uclaNew, df.amazNew, 'wo')
plt.xlabel('UCLA Store Price')
plt.ylabel('Amazon Price')

plt.hist(df['uclaNew'] - df['amazNew'])
plt.xlabel('Bookstore price - Amazon price')
plt.ylabel('Count')

ucla_mean = df['uclaNew'].mean()
ucla_var = df['uclaNew'].var()

amaz_mean = df['amazNew'].mean()
amaz_var = df['amazNew'].var()

diff_mean = df['diff'].mean()
diff_std = df['diff'].std()

print ucla_mean, ucla_var
print amaz_mean, amaz_var
print diff_mean, diff_std

SE = diff_std / np.sqrt(len(df))
T = (diff_mean - 0.0) / SE
T

from scipy.stats import t
2 * (1.0 - t.cdf(T, len(df) - 1))

from scipy.stats import ttest_rel
T, p_value = ttest_rel(df['uclaNew'], df['amazNew'])
print T, p_value

