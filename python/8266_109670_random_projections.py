get_ipython().magic('matplotlib inline')
#%load_ext autoreload
#%autoreload 2
get_ipython().magic('reload_ext autoreload')
import numpy as np
import matplotlib.pyplot as plt
import math, sys, os
from numpy.random import randn
from sklearn.datasets import make_blobs

# setup pyspark for IPython_notebooks
spark_home = os.environ.get('SPARK_HOME', None)
sys.path.insert(0, spark_home + "/python")
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.8.2.1-src.zip'))
execfile(os.path.join(spark_home, 'python/pyspark/shell.py'))

data_home = os.environ.get('DATA_HOME', None)
sys.path.insert(0, data_home)

# data
from gen_data import make_blobs_rdd

# utilitiy functions for this notebook
from lsh_util import *

# make some data
N = 1000
d = 2
k = 5
sigma = 3
bound = 10

data_RDD = make_blobs_rdd(N, d, k, sigma, bound, sc)
data_RDD.take(2)

def config_random_projection(d, n_hyperplanes = 5, scale = 2.0, seed = None):    
    # random projection vectors
    Z = (np.random.rand(d, n_hyperplanes) - 0.5) * scale
    def projection_func(tup):
        y, x = tup # expect key (int, 1xD vector)
        projs = x.T.dot(Z) # random projections
        bucket = to_bucket(projs)
        return (bucket, y)
    
    return (Z,projection_func)

Z, hash_func = config_random_projection(d)

gini_impurities = data_RDD.map(hash_func).map(to_dict).reduceByKey(reduce_count_clusters).map(gini_impurity).collect()
for b, g, c in sorted(gini_impurities):
    print "bucket: %s , in bucket: %d , gini_impurity: %f" % (b, c, g)

c0 = np.stack(data_RDD.filter(lambda t: t[0] == 0).map(lambda t: t[1]).collect())
c1 = np.stack(data_RDD.filter(lambda t: t[0] == 1).map(lambda t: t[1]).collect())
c2 = np.stack(data_RDD.filter(lambda t: t[0] == 2).map(lambda t: t[1]).collect())
c3 = np.stack(data_RDD.filter(lambda t: t[0] == 3).map(lambda t: t[1]).collect())
c4 = np.stack(data_RDD.filter(lambda t: t[0] == 4).map(lambda t: t[1]).collect())

plt.scatter(c0[:,0],c0[:,1],color='g')
plt.scatter(c1[:,0],c1[:,1],color='y')
plt.scatter(c2[:,0],c2[:,1],color='b')
plt.scatter(c3[:,0],c3[:,1],color='k')
plt.scatter(c4[:,0],c4[:,1],color='m')

# projection vectors
plt.scatter(Z.T[:,0],Z.T[:,1],color='r',s=50)

# impurity as we scale up the number of hyperplanes used for projections

for n_Z in range(10,201,10):
    Z, hash_func = config_random_projection(d, n_Z)
    gini_impurities = data_RDD.map(hash_func).map(to_dict).reduceByKey(reduce_count_clusters).map(gini_impurity).collect()
    g_i = weighted_gini(gini_impurities)
    print "%d projections, gini_impurity: %f" % (n_Z, g_i)
    



