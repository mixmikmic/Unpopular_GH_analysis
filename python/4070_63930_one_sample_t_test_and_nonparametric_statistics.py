import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

from scipy.stats import norm
population = norm.rvs(loc=2, scale=0.25, size=1000)
n, bins, patches = plt.hist(population, bins=20)

from scipy.stats import probplot
(osm, osr), (slope, intercept, r) = probplot(population, plot=plt)

import random
sample = np.array(random.sample(population, k=25))

from scipy.stats import ttest_1samp
T, p_value = ttest_1samp(sample, population.mean())
T, p_value

from scipy.stats import t
SE = sample.std(ddof=1) / np.sqrt(sample.size)
T = (sample.mean() - population.mean()) / SE
p_value = 2 * t.cdf(-abs(T), df=sample.size - 1)
print T, p_value

import pandas as pd
df = pd.read_csv('textbooks.txt', sep='\t')
from scipy.stats import wilcoxon
T, p_value = wilcoxon(df['uclaNew'], df['amazNew'])
T, p_value

