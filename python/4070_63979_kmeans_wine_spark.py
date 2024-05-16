import numpy as np
from pyspark.mllib.clustering import KMeans

str_lines = sc.textFile('/Users/jhalverson/data_science/machine_learning/wine.csv')
data_features = str_lines.map(lambda line: np.array([float(x) for x in line.split(',')[1:]]))
data_features.take(2)

from pyspark.mllib.feature import StandardScaler
stdsc = StandardScaler(withMean=True, withStd=True).fit(data_features)
data_features_std = stdsc.transform(data_features)
data_features_std.take(3)

from pyspark.mllib.stat import Statistics
data_features_std_stats = Statistics.colStats(data_features_std)
print 'means:', data_features_std_stats.mean()
print 'variances:', data_features_std_stats.variance()

np.set_printoptions(precision=2, linewidth=100)
from pyspark.mllib.stat import Statistics
print Statistics.corr(data_features_std, method='pearson')

def error(point, model):
    center = model.centers[clusters.predict(point)]
    return np.sqrt(sum([x**2 for x in (point - center)]))

errors =  []
k_clusters = range(1, 11)
for k in k_clusters:
    clusters = KMeans.train(data_features_std, k=k, runs=25, initializationMode="k-means||")
    WSSSE = data_features_std.map(lambda point: error(point, clusters)).reduce(lambda x, y: x + y)
    errors.append(WSSSE)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

plt.plot(k_clusters, errors, 'k-', marker='o', mfc='w')
plt.xlabel('k')
plt.ylabel('Distortion')

