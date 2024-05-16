import pandas as pd
import matplotlib as plt

#jupyter magic so the plots are displayed inline
get_ipython().run_line_magic('matplotlib', 'inline')

recent_grads = pd.read_csv('recent-grads.csv')
recent_grads.iloc[0]

recent_grads.head(1)

recent_grads.tail(1)

recent_grads.describe()

recent_grads = recent_grads.dropna()
recent_grads

recent_grads.plot(x='Sample_size', y='Median', kind = 'scatter')
recent_grads.plot(x='Sample_size', y='Unemployment_rate', kind = 'scatter')
recent_grads.plot(x='Full_time', y='Median', kind = 'scatter')
recent_grads.plot(x='ShareWomen', y='Unemployment_rate', kind = 'scatter')
recent_grads.plot(x='Men', y='Median', kind = 'scatter')
recent_grads.plot(x='Women', y='Median', kind = 'scatter')

recent_grads['Median'].hist(bins=25)

recent_grads['Employed'].hist(bins=25)

recent_grads['Full_time'].hist(bins=25)

recent_grads['ShareWomen'].hist(bins=25)

recent_grads['Unemployment_rate'].hist(bins=25)

recent_grads['Men'].hist(bins=25)

recent_grads['Women'].hist(bins=25)

from pandas.plotting import scatter_matrix

scatter_matrix(recent_grads[['Sample_size', 'Median']], figsize=(10,10))

scatter_matrix(recent_grads[['Men', 'ShareWomen', 'Median']], figsize=(10,10))

recent_grads[:10].plot(kind='bar', x='Major', y='ShareWomen', colormap='winter')
recent_grads[163:].plot(kind='bar', x='Major', y='ShareWomen', colormap='winter')

recent_grads[:10].plot(kind='bar', x='Major', y='Median', colormap='winter')
recent_grads[163:].plot(kind='bar', x='Major', y='Median', colormap='winter')



