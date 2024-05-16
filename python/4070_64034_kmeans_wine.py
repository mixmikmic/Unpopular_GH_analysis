import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

df = pd.read_csv('wine.csv', header=None)
df.head()

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_std = stdsc.fit_transform(df.iloc[:, 1:].values)
y = df.iloc[:, 0].values - 1

np.set_printoptions(precision=2, linewidth=100)
print X_std[:5]

from sklearn.cluster import KMeans

k_range = range(1, 11)
distortions = []
for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=25)
    km.fit(X_std)
    distortions.append(km.inertia_)
plt.plot(k_range, distortions, 'k-', marker='o', mfc='w')
plt.xlabel('k')
plt.ylabel('Distortion')

from sklearn.metrics import accuracy_score

km = KMeans(n_clusters=3, init='k-means++', n_init=25, random_state=0)
accuracy_score(km.fit_predict(X_std), y[::-1])

