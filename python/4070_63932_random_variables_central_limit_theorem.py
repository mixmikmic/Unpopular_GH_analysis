outcomes = [1, 2, 3, 4, 5, 6]
weights = [1/6.0 for _ in range(6)]
E_die = sum(outcome * weight for outcome, weight in zip(outcomes, weights))
print E_die

var = sum(weight * (outcome - E_die)**2 for outcome, weight in zip(outcomes, weights))
print var, var**0.5

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from scipy.stats import lognorm
x = np.linspace(0.01, 10, num=100)
samples = lognorm.rvs(s=0.75, size=1000)
plt.plot(x, lognorm.pdf(x, s=0.75))

lognorm.mean(s=0.75), lognorm.std(s=0.75)

f, (ax0, ax1, ax2, ax3) = plt.subplots(4)
for i, n in enumerate([1, 10, 100, 1000]):
    z = []
    for _ in range(10000):
        z.append(np.mean(lognorm.rvs(s=0.75, size=n)))
    print np.mean(z), np.std(z), lognorm.std(s=0.75)/n**0.5
    exec('ax' + str(i) + '.hist(z, bins=100, range=(0, 5))')

from scipy.stats import uniform
samples = uniform.rvs(loc=0.0, scale=10.0, size=10000)
n, bins, patches = plt.hist(samples, bins=25)

from scipy.stats import uniform
f, (ax0, ax1, ax2, ax3) = plt.subplots(4)
for i, n in enumerate([1, 10, 100, 1000]):
    z = []
    for _ in range(10000):
        z.append(np.mean(uniform.rvs(loc=0.0, scale=1.0, size=n))) # range is [loc, loc + scale)
    print np.mean(z), np.std(z), uniform.std(loc=0.0, scale=1.0)/n**0.5
    exec('ax' + str(i) + '.hist(z, bins=100, range=(0, 5))')

