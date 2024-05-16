import multiprocess as mp

from glob import glob
import re
import pandas as pd
import numpy as np
import dill
import os
import warnings
import h5py

import cv2

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise
#from scipy.sparse import csr_matrix, vstack
#from sklearn.feature_extraction.text import TfidfTransformer

# The data set of the higher resolution images
large_images = pd.read_pickle('../priv/pkl/20_wine_label_analysis_large_labels.pkl')
large_images.shape

# All remaining images
all_images = pd.read_pickle('../priv/pkl/20_wine_label_analysis_all_labels.pkl')
mask = all_images['basename'].isin(large_images['basename']).pipe(np.invert)
all_images = all_images.loc[mask]
all_images.shape

is_trial = True

st = pd.HDFStore('../priv/data/features.h5', 'r')

mask = st['basename'].isin(large_images.basename)

print('Total images: {}'.format(st['basename'].shape[0]))
print('Total features: {}'.format(st['index']['end'].max()))

if is_trial:
    max_index = st['index'].loc[mask,'end'].max()
else:
    max_index = st['index']['end'].max()
    
print('Maximum index: {}'.format(max_index))

st.close()

def mini_batch_kmeans(data_path, out_path, max_index, n_clusters, frac_points=0.5):
    
    print n_clusters
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Select randomized indexes for data and read it in
    st = pd.HDFStore(data_path, 'r')
    n_points = int(frac_points * max_index)
    indexes = np.random.choice(np.arange(max_index), n_points, replace=False)
    data = st['features'].loc[indexes].values
    
    
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, init_size=3*n_clusters)
    model.fit(data)
    
    st.close()
    
    # Write the resulting model clusters out to a file
    st = h5py.File(out_path, 'a')
    if str(n_clusters) in st.keys():
        st.pop(str(n_clusters))
        
    mod = st.create_dataset(str(n_clusters), model.cluster_centers_.shape)
    mod[:,:] = model.cluster_centers_
    
    st.close()
    
    with open('../priv/models/minibatch_kmeans_clusters_{}.pkl'.format(n_clusters),'wb') as fh:
        dill.dump(model.cluster_centers_, fh)
        
    return

nclusters = [1500, 1536, 2000, 2500, 3000, 5000]

for cluster in nclusters:
    mini_batch_kmeans('../priv/data/features.h5', '../data/kmeans.h5', max_index, cluster)

get_ipython().system(' echo "pushover \'kmeans clustering finished\'" | /usr/bin/zsh')

