(14/15.0) * (13/14.0) * (12/13.0)

from scipy.special import binom

total_number_of_groups_of_three = binom(15, 3)
number_of_groups_of_three_excluding_one_person = binom(1, 0) * binom(14, 3)
print number_of_groups_of_three_excluding_one_person / total_number_of_groups_of_three

import random

trials = 10000
success = 0
for _ in range(trials):
    tickets = range(1, 31)
    for _ in range(7):
        value = random.choice(tickets)
        if (value == 1): success += 1
        tickets.remove(value)
print success / float(trials)

1.0 - (29/30.0) * (28/29.0) * (27/28.0) * (26/27.0) * (25/26.0) * (24/25.0) * (23/24.0)

trials = 10000
success = 0
tickets = range(1, 31)
for _ in range(trials):
    for _ in range(7):
        value = random.choice(tickets)
        if (value == 1):
            success += 1
            break
print success / float(trials)

1.0 - (29/30.0)**7

from scipy.stats import binom
1.0 - binom.cdf(k=0, n=7, p=1/30.0)

from scipy.special import binom as choose
sum([choose(7, k) * ((1/30.)**k) * (29/30.)**(7-k) for k in range(1, 8)])

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

draws = 7
plt.vlines(x=range(draws + 1), ymin=np.zeros(draws + 1), ymax=binom.pmf(range(draws + 1), n=draws, p=1/30.0), lw=4)
plt.xlabel('Raffle drawing index')
plt.ylabel('Probability')
plt.xlim(-0.5, 7)
plt.ylim(0, 1)

