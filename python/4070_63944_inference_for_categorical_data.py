import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['font.size'] = 14

from scipy.stats import bernoulli as bern
population = bern.rvs(p=0.82, size=100000)

def proportion(x):
    x = np.asarray(x)
    return sum(x) / float(x.size)

import random
def sample_mean_std(population, sample_size):
    """Return the mean and standard deviation for a
       sample of the populations."""
    s = np.array(random.sample(population, sample_size))
    return s.mean(), s.std(ddof=1)

p_population = proportion(population)
print p_population

samples = 1000
pairs = [sample_mean_std(population, sample_size=50) for _ in range(samples)]
means, stds = zip(*pairs)
print sum(means) / len(means), np.array(means).std()

n, bins, patches = plt.hist(means, bins=15, align='mid')
plt.xlabel("Sample mean")
plt.ylabel("Count")

print np.array(means).std(), population.std() / np.sqrt(50), np.sqrt(p_population * (1.0 - p_population) / 50)

p_hat = proportion(random.sample(population, 50))
print p_hat

np.sqrt(p_hat * (1.0 - p_hat) / 50)

from scipy.stats import norm
SE = np.sqrt(0.5 * (1.0 - 0.5) / 1028)
z_score = (0.56 - 0.5) / SE
p_value = 1.0 - norm.cdf(z_score)
print SE, z_score, p_value, p_value > 0.05

p_trmt = 500 / float(500 + 44425)
p_ctrl =  505 / float(505 + 44405)
p_hat = (500 + 505) / float(500 + 44425 + 505 + 44405)
diff = p_trmt - p_ctrl
print p_trmt, p_ctrl, p_hat, diff

p_hat * 44925 > 10, (1.0 - p_hat) * 44925 > 10, p_hat * 44910 > 10, (1.0 - p_hat) * 44910 > 10

SE = np.sqrt(p_trmt * (1.0 - p_trmt) / (500 + 44425) + p_ctrl * (1.0 - p_ctrl) / (505 + 44405))
SE_ = np.sqrt(p_hat * (1.0 - p_hat) / (500 + 44425) + p_hat * (1.0 - p_hat) / (505 + 44405))
print SE, SE_

diff + 1.65 * SE, diff - 1.65 * SE

z_score = (diff - 0.0) / SE_
p_value = 2 * norm.cdf(z_score)

print z_score, p_value, p_value > 0.05

p_curr = 899 / 1000.0
p_pros = 958 / 1000.0
diff = p_pros - p_curr
print diff

SE = np.sqrt(p_curr * (1.0 - p_curr) / 1000 + p_pros * (1.0 - p_pros) / 1000) 
print SE

z_score = (diff - 0.03) / SE
print z_score

p_value = 1.0 - norm.cdf(z_score)
print p_value, p_value > 0.05

