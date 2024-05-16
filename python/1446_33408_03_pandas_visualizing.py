import pandas as pd

df = pd.read_csv('http://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/diamonds.csv')
df.head()

df = df.drop(['Unnamed: 0'], axis = 1)

df.info()

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(x='carat', y='depth', data=df, c='b', alpha=.10)

df.plot.scatter(x='carat', y='price')

import seaborn as sns

sns.countplot(x='cut', data=df)
sns.despine()

sns.barplot(x='cut', y='price', data=df)

g = sns.FacetGrid(df, col='color', hue='color', col_wrap=4)
g.map(sns.regplot, 'carat', 'price')

