import numpy as np
from numpy.random import lognormal
from numpy.random import choice

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

population = lognormal(mean=1.0, sigma=0.5, size=10000)
population.mean(), population.std()

plt.hist(population, bins=25)

sample = choice(population, size=25, replace=False)
sample.mean(), sample.std()

sample_statistics = []
for _ in range(1000):
     sample_statistics.append(choice(population, size=25, replace=False).mean())

plt.hist(sample_statistics)

np.array(sample_statistics).mean(), np.array(sample_statistics).std()

np.array(sample_statistics).std() * 25**0.5

np.array(sample_statistics).std()

sample.std() / 25**0.5

bstraps = []
for _ in range(1000):
     bstraps.append(choice(sample, size=25, replace=True).mean())

plt.hist(bstraps)

np.array(bstraps).std()

np.array(bstraps).mean()

for i, val in enumerate(sorted(bstraps)):
     if i == 49 or i == 950: print i, val

np.percentile(bstraps, q=5), np.percentile(bstraps, q=95)

from scipy.stats import t
lo = sample.mean() + t.ppf(0.05, df=24) * sample.std() / 25**0.5
hi = sample.mean() - t.ppf(0.05, df=24) * sample.std() / 25**0.5
lo, hi

