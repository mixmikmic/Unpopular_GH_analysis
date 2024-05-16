import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000)

plt.scatter(pageSpeeds, purchaseAmount)
np.cov(pageSpeeds, purchaseAmount)

purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds
plt.scatter(pageSpeeds, purchaseAmount)
np.cov(pageSpeeds, purchaseAmount)

purchaseAmount = 100 - pageSpeeds * 3
plt.scatter(pageSpeeds, purchaseAmount)
np.corrcoef(pageSpeeds, purchaseAmount)

