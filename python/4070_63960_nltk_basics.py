from nltk.book import *

type(text1)

text1.concordance('biscuit')

text1[:50]

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

text1.dispersion_plot(['whale', 'white', 'sea'])

len(text1)

len(set(text1)) / float(len(text1))



