import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

from scipy.stats import norm
normal_sample = norm.rvs(size=10000)

x = np.linspace(-5, 5, num=250)
plt.plot(x, norm.pdf(x), 'k-')
plt.xlabel('x')
plt.ylabel('P(x)')

plt.plot(x, norm.cdf(x), 'k-')
plt.xlabel('x')
plt.ylabel('CDF(x)')

plt.hist(norm.cdf(normal_sample), bins=20)
plt.xlabel('CDF')
plt.ylabel('Count')

from scipy.stats import uniform
uniform_sample = uniform.rvs(size=10000)

plt.hist(norm.ppf(uniform_sample), bins=20)
plt.xlabel('CDF(x)')
plt.ylabel('Count')

