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

'''
load in trial data
'''
columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked','Right Reward Prob','Left Reward Prob','Reward Given']

root_dir = '/Users/shayneufeld/GitHub/mouse_bandit/data/70_30_trial_data'

trial_df = []
mouse_ids = []
session_ids = []
for file in os.listdir(root_dir):
    if not file[0] == '.':
        file_name = os.path.join(root_dir,file)
        trial_df.append(pd.read_csv(file_name,names=columns))
        mouse_ids.append(file[:file.index('_')])
        session_ids.append(file[file.index('_')+1:file.index('_')+7])

for i,df in enumerate(trial_df):
    
    curr_feature_matrix = bp.create_feature_matrix(df,10,mouse_ids[i],session_ids[i],feature_names='Default')
    
    if i == 0:
        master_matrix = curr_feature_matrix.copy()
    else:
        master_matrix = master_matrix.append(curr_feature_matrix)
    

master_matrix.shape

master_matrix.head(30)

def encode_categorical(array):
    if not (array.dtype == np.dtype('float64') or array.dtype == np.dtype('int64')) :
        return preprocessing.LabelEncoder().fit_transform(array) 
    else:
        return array

categorical = (master_matrix.dtypes.values != np.dtype('float64'))
master_matrix_1hot = master_matrix.apply(encode_categorical)

# Apply one hot endcoing
encoder = OneHotEncoder(categorical_features=categorical, sparse=False)  # Last value in mask is y
x = encoder.fit_transform(master_matrix.values)

master_matrix.to_csv(os.path.join(root_dir,'master_7030_df.csv'))

master_matrix

