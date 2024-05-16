import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

columns = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',            'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',            'OD280/OD315 of diluted wines', 'Proline']
df = pd.read_csv('wine.csv', names=columns)
df.head()

xf = df.iloc[:,1:]
xf = (xf - xf.mean()) / xf.std()
xf.head(3)

xf.describe().applymap(lambda x: round(x, 1))

xf.skew()

xf.kurt()

for column in xf.columns:
     plt.hist(xf[column], histtype='step')

plt.hist(xf.Alcohol)

from scipy.stats import anderson
a2, crit, sig = anderson(xf.Alcohol, 'norm')
a2, crit, sig

plt.hist(xf['Malic acid'])

a2, crit, sig = anderson(xf['Malic acid'], 'norm')
a2, crit, sig

plt.hist(xf.Magnesium)

for column in xf.columns:
     a2, crit, sig = anderson(xf[column], 'norm')
     print column, '%.2f' % a2, a2 < crit[4]

xf[df['class'] == 1].cov().applymap(lambda x: round(x, 1))

xf[df['class'] == 2].cov().applymap(lambda x: round(x, 1))

xf[df['class'] == 3].cov().applymap(lambda x: round(x, 1))

X = df.iloc[:,1:].values
y = df['class'].values

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_std_lda = lda.fit_transform(X_std, y)

plt.scatter(X_std_lda[y==1, 0], X_std_lda[y==1, 1])
plt.scatter(X_std_lda[y==2, 0], X_std_lda[y==2, 1])
plt.scatter(X_std_lda[y==3, 0], X_std_lda[y==3, 1])
plt.xlabel('LD 1')
plt.ylabel('LD 2')

