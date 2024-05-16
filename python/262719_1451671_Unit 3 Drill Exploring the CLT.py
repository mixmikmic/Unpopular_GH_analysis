import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

pop1 = np.random.binomial(10, 0.2, 10000)
pop2 = np.random.binomial(10, 0.5, 10000)

# Make a histogram for two groups. 

plt.hist(pop1, alpha=0.5, label='Population 1')
plt.hist(pop2, alpha=0.5, label='Population 2')
plt.legend(loc='upper right')
plt.show()

#Populations are not normal

sample1 = np.random.choice(pop1, 1000, replace=True)
sample2 = np.random.choice(pop2, 1000, replace=True)

plt.hist(sample1, alpha=0.5, label='sample 1')
plt.hist(sample2, alpha=0.5, label='sample 2')
plt.legend(loc='upper right')

plt.show()

print(sample1.mean())
print(sample2.mean())
print(sample1.std())
print(sample2.std())

sample3 = np.random.choice(pop1, 20, replace=True)
sample4 = np.random.choice(pop2, 20, replace=True)

plt.hist(sample3, alpha=0.5, label='sample 1')
plt.hist(sample4, alpha=0.5, label='sample 2')
plt.legend(loc='upper right')

plt.show()

print(sample3.mean())
print(sample4.mean())
print(sample3.std())
print(sample4.std())

# Problem 2 p = 0.3
pop1 = np.random.binomial(10, 0.3, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 

sample1 = np.random.choice(pop1, 100, replace=True)
sample2 = np.random.choice(pop2, 100, replace=True)

from scipy.stats import ttest_ind
print(ttest_ind(sample2, sample1, equal_var=False))

#p = 0.4
pop1 = np.random.binomial(10, 0.4, 10000)
pop2 = np.random.binomial(10,0.5, 10000) 


sample1 = np.random.choice(pop1, 100, replace=True)
sample2 = np.random.choice(pop2, 100, replace=True)

from scipy.stats import ttest_ind
print(ttest_ind(sample2, sample1, equal_var=False))

pop1 = np.random.geometric(0.2, 10000)
pop2 = np.random.geometric(0.5, 10000)

# Make a histogram for two groups. 

plt.hist(pop1, alpha=0.5, label='Population 1')
plt.hist(pop2, alpha=0.5, label='Population 2')
plt.legend(loc='upper right')
plt.show()

#Populations are not normal

sample1 = np.random.choice(pop1, 1000, replace=True)
sample2 = np.random.choice(pop2, 1000, replace=True)

plt.hist(sample1, alpha=0.5, label='sample 1')
plt.hist(sample2, alpha=0.5, label='sample 2')
plt.legend(loc='upper right')

plt.show()

