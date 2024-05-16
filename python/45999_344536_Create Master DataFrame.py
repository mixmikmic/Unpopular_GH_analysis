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

db[db['Session ID']=='03262017_K13']

conditions = ['100-0','90-10','80-20','70-30']
probs = [1,0.9,0.8,0.7]

for condition,prob in zip(conditions,probs):
    print(condition)
    print(prob)

master_matrix = np.zeros(1)

conditions = ['100-0','90-10','80-20','70-30']
probs = [1,0.9,0.8,0.7]

for condition,prob in zip(conditions,probs):
    r = db[((db['Left Reward Prob'] == prob) |  (db['Right Reward Prob'] == prob))].copy()
    r = r[r['p(high Port)'] > prob-0.1].copy()
    r = r[r['Block Range Min'] == 50].copy()
    r['Condition'] = condition
    session_names = r['Session ID'].values
    conditions_ = r['Condition'].values
    
    '''
    load in trial data
    '''
    columns = ['Elapsed Time (s)','Since last trial (s)','Trial Duration (s)','Port Poked','Right Reward Prob','Left Reward Prob','Reward Given']

    root_dir = '/Users/shayneufeld/GitHub/mouse_bandit/data/trial_data'

    trial_df = []

    #load in trials
    for session in session_names:
        full_name = session + '_trials.csv'

        path_name = os.path.join(root_dir,full_name)

        trial_df.append(pd.read_csv(path_name,names=columns))

    mouse_ids = r['Mouse ID'].values
    
    
    #create feature matrix
    for i,df in enumerate(trial_df):
    
        curr_feature_matrix = bp.create_feature_matrix(df,10,mouse_ids[i],session_names[i],feature_names='Default')
        curr_feature_matrix['Condition'] = condition

        if master_matrix.shape[0]==1:
            master_matrix = curr_feature_matrix.copy()
        else:
            master_matrix = master_matrix.append(curr_feature_matrix)

master_matrix.index = np.arange(master_matrix.shape[0])

master_matrix[master_matrix['Session ID']=='03242017_K13'].index.values

m=master_matrix.copy()

m = m.drop(master_matrix[master_matrix['Session ID']=='03242017_K13'].index.values)

master_matrix = m.copy()

master_matrix.to_csv(os.path.join(root_dir,'master_data.csv'))

root_dir

