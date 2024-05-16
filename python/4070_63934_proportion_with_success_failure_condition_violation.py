import random
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['font.size'] = 14

p_null = 0.1
trials = 10000
data = []
for _ in range(trials):
    complications = 0
    for _ in range(100):
        if random.random() < p_null: complications += 1
    data.append(complications)
from collections import Counter
cnt = Counter(data)
plt.vlines(x=cnt.keys(), ymin=np.zeros(len(cnt)), ymax=cnt.values(), lw=4)
plt.xlabel('Number of complications out of 100')
plt.ylabel('Count')

p_hat = 3 / 62.0
print sum([1 for d in data if d / 100.0 <= p_hat]) / float(trials)

from scipy.stats import binom
plt.vlines(range(100), ymin=np.zeros(100), ymax=binom.pmf(k=range(100), p=0.1, n=100), lw=4)
plt.xlim(0, 25)
plt.xlabel('Number of complications out of 100')
plt.ylabel('Probability')

binom.cdf(k=int(100 * p_hat), p=0.1, n=100)

def sizes_cdfs(p_thres):
    sizes = [10**n for n in range(1, 8)]
    cdf = [binom.cdf(k=int(p_thres * size), p=0.1, n=size) for size in sizes]
    return sizes, cdf

plt.loglog(*sizes_cdfs(p_hat), label=r'$p\leq 0.048$')
plt.loglog(*sizes_cdfs(0.096), label=r'$p\leq 0.096$')
#plt.loglog(sizes_cdfs(p_hat)[0], sizes_cdfs(p_hat)[1], label=r'$p\leq 0.048$')
#plt.loglog(sizes_cdfs(0.096)[0], sizes_cdfs(0.096)[1], label=r'$p\leq 0.096$')
plt.xlabel('Size')
plt.ylabel('p-value')
plt.legend(loc='upper right')
plt.ylim(1e-4, 1)

