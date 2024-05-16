# works for python 3.0
from IPython.core.display import HTML
import urllib.request
request = urllib.request.Request('http://bit.ly/1Bf5Hft')
response = urllib.request.urlopen(request)
HTML(response.read().decode('utf-8'))

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

data = pd.read_csv('stroopdata.csv')

data.info()

data.describe()

data.plot(kind='kde')

data.boxplot()

s1 = data['Congruent']
s2 = data['Incongruent']
mean_diff = s2.mean()-s1.mean()
length = len(s1)
sample_sd = np.sqrt(np.var(s2-s1)*length/(length-1))
t = mean_diff/sample_sd*np.sqrt(length)
print('mean_diff={0:.2f}\n sample_sd={1:.3f}\n t={2:.3f}'.format(mean_diff,sample_sd,t))



