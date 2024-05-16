import pandas as pd
import numpy as np
import os
import h5py
import dill
import warnings

warnings.filterwarnings('ignore')

hist_file = '../priv/data/hist.h5'
hs = h5py.File(hist_file, 'r')

features_file = '../priv/data/features.h5'
ft = pd.HDFStore(features_file, 'r')
files = ft['image_path'].values
ft.close()

index_file = '../priv/data/inverted_index.h5'
ix = pd.HDFStore(index_file, 'w')

for ncluster in hs.keys():
        
    print ncluster
    # Get the histograms for a given number of clusters
    dat = hs[ncluster]
    
    df_list = list()
    for clust in range(dat.shape[1]):
        
        # All files that have data at a given cluster
        file_index = np.where(dat[:, clust] !=0 )[0]
        clust_counts = dat[:, clust][file_index].astype(int)
        df = pd.DataFrame({'file':files[file_index], 
                           'count':clust_counts},
                          index = pd.Index([clust] * len(file_index)))
        df_list.append(df)
        
    cluster_df = pd.concat(df_list)
    ix.append(ncluster, cluster_df)
        
hs.close()
ix.close()

