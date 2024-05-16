import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
from sklearn import preprocessing
import sys
import os
get_ipython().magic('matplotlib inline')

#load in data base
db = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/session_record.csv',index_col=0)
session_names = db['Session ID'].values
mouse_ids = db['Mouse ID'].values
session_ids = db['Session ID'].values

'''
load in trial data
'''
columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked','Right Reward Prob','Left Reward Prob','Reward Given']

root_dir = '/Users/shayneufeld/GitHub/mouse_bandit/data/trial_data'
save_dir = '/Users/shayneufeld/GitHub/mouse_bandit/data/feature_data'

for i,session in enumerate(session_names):
    #get name of trial csv
    full_name = session + '_trials.csv'
    #get name of full patch to trial data
    path_name = os.path.join(root_dir,full_name)
    #read in trial csv file
    trial_df = pd.read_csv(path_name,names=columns)
    #create reduced feature matrix
    feature_matrix = bp.create_reduced_feature_matrix(trial_df,mouse_ids[i],session_ids[i],feature_names='Default')
    #create file name to be saved
    save_name = session + '_features.csv'
    #save feature matrix for this session
    feature_matrix.to_csv(os.path.join(save_dir,save_name))

