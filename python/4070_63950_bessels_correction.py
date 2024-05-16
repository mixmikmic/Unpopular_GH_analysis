import random
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['font.size'] = 14
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['lines.markersize'] = 8

random.seed(1234)
np.random.seed(1234)

population = np.random.random_integers(low=0, high=100, size=10000)
mu = population.mean()
s2 = population.var(ddof=0)
print mu, s2

def var(v, mu, df=0):
    """Compute the variance of a list using the supplied average."""
    u = v - mu
    u *= u
    return u.sum() / (u.size - df)

var_with_population_mean = []
var_with_sample_mean = []
var_with_sample_mean_bessel = []
for _ in range(20):
    sample = np.array(random.sample(population, k=10))
    var_with_population_mean.append(var(sample, mu, 0))
    var_with_sample_mean.append(sample.var(ddof=0))
    var_with_sample_mean_bessel.append(sample.var(ddof=1))

plt.figure(figsize=(9, 6))
plt.plot(var_with_population_mean, 'wo', label='population mean')
plt.plot(var_with_sample_mean, 'r^', label='sample mean')
plt.plot(var_with_sample_mean_bessel, 'b|', label='sample mean w/ Bessel')
plt.plot([0, 20], [s2, s2], 'k:')
plt.xlabel('index')
plt.ylabel('variance')
plt.legend(loc='upper right', fontsize=12)

print np.array(var_with_population_mean).mean() / s2
print np.array(var_with_sample_mean).mean() / s2
print np.array(var_with_sample_mean_bessel).mean() / s2

print sum((np.array(var_with_sample_mean) - s2)**2)
print sum((np.array(var_with_sample_mean_bessel) - s2)**2)

