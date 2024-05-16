import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

x = norm.rvs(loc=2, scale=1, size=100)
plt.hist(x, normed=True, width=0.4)

kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(x[:,np.newaxis])
r = np.linspace(-2, 7, num=100)
density = np.exp(kde.score_samples(r[:,np.newaxis]))
plt.hist(x, normed=True, width=0.4)
plt.plot(r, density, 'r-')

