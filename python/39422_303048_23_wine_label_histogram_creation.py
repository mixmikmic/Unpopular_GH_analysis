import pandas as pd
import numpy as np
import os
import h5py
import warnings

from sklearn.metrics.pairwise import euclidean_distances
#from scipy.sparse import csr_matrix, vstack
# from sklearn.feature_extraction.text import TfidfTransformer

# The data set of the higher resolution images
large_images = pd.read_pickle('../priv/pkl/20_wine_label_analysis_large_labels.pkl')
large_images.shape

is_trial = True

warnings.filterwarnings('ignore')

kmeans_file = '../priv/data/kmeans.h5'
km = h5py.File(kmeans_file, 'r')

features_file = '../priv/data/features.h5'
ft = pd.HDFStore(features_file, 'r')

hist_file = '../priv/data/hist.h5'
hs = pd.HDFStore(hist_file, 'w')

if is_trial:
    mask = ft['basename'].isin(large_images.basename)
    max_index = ft['index'].loc[mask,'end'].max()
    nimages = mask.sum()
else:
    max_index = ft['index']['end'].max()
    nimages = ft['index'].shape[0]
    
    
# for ncluster in ['1500']:
for ncluster in km.keys():
    print(ncluster)
    
    km_matrix = km[ncluster].value

    hist_list = list()
    
    for im in range(nimages):

        indexes = ft['index'].iloc[im]
        image_path = ft['image_path'].iloc[im]
        
        # This is a much faster and lower memory way of accessing a subset
        # of a dataframe
        features = ft.select('features', start=indexes.beg, stop=indexes.end).values
        
        # Pairwise euclidean distances
        ec = euclidean_distances(features, km_matrix)
        
        # Closest cluster id and count
        closest_clust_id = np.argmin(ec, axis=1)
        cluster_id, word_count = np.unique(closest_clust_id, return_counts=True)
        
        # Dense matrix of word counts
        bag_of_nums = np.zeros(int(ncluster), dtype=np.int)
        bag_of_nums[cluster_id] = word_count            
        
        # Store the histogram in the proper row
        hist_list.append(pd.Series(bag_of_nums, name=image_path))
        
    hist_df = pd.concat(hist_list, axis=1).T
    hist_df = hist_df.reset_index().rename(columns={'index':'image_path'})
    hs.append(ncluster, hist_df)
        
        
km.close()
ft.close()
hs.close()

