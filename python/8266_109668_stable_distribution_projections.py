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
k = 3
sigma = 1
bound = 10*sigma

data_RDD = make_blobs_rdd(N, d, k, sigma, bound, sc)
data_RDD.take(2)

def config_stable_dist_proj(d, p = 5, r = 2.0, seed = None):
    # random projection vectors
    A = np.random.multivariate_normal(np.zeros(d), np.eye(d), p)
    B = np.random.rand(1,p)
    def projection_func(tup):
        y, x = tup # expect key (int, 1xD vector)
        projs = ((A.dot(x) / r) + B).flatten()
        bucket = to_bucket(projs)
        
        return (bucket, y)
    
    return (A, B, projection_func)

A, B, hash_func = config_stable_dist_proj(d)

gini_impurities = data_RDD.map(hash_func).map(to_dict).reduceByKey(reduce_count_clusters).map(gini_impurity).collect()
for b, g, c in sorted(gini_impurities):
    print "bucket: %s , in bucket: %d , gini_impurity: %f" % (b, c, g)

# impurity as we scale up the number of hyperplanes used for projections
for n_Z in range(10,201,10):
    A, B, hash_func = config_stable_dist_proj(d, n_Z)
    gini_impurities = data_RDD.map(hash_func).map(to_dict).reduceByKey(reduce_count_clusters).map(gini_impurity).collect()
    g_i = weighted_gini(gini_impurities)
    print "%d projections, gini_impurity: %f" % (n_Z, g_i)
    



