import random
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

p = 0.65

from scipy.stats import beta
x = np.linspace(0.0, 1.0, num=250)
plt.plot(x, beta.pdf(x, 50, 50), 'k-')
plt.plot([0.5, 0.5], [0, 15], 'k:')
plt.ylim(0, 15)
plt.xlabel(r'$p$')
plt.ylabel(r'$P(p)$')

import random

heads = 0
tails = 0
flips = 100
for _ in xrange(flips):
    if (random.random() < p):
        heads += 1
    else:
        tails += 1
print heads / float(flips)

plt.plot(x, beta.pdf(x, 50, 50), 'k-', label='prior')
plt.plot(x, beta.pdf(x, 50 + heads, 50 + tails), 'r-', label='posterior')
plt.plot([0.5, 0.5], [0, 15], 'k:')
plt.ylim(0, 15)
plt.legend(loc='upper right')
plt.xlabel(r'$p$')
plt.ylabel(r'$P(p)$')

1.0 - beta.cdf(0.49, 50 + heads, 50 + tails) - (1.0 - beta.cdf(0.51, 50 + heads, 50 + tails)) 

