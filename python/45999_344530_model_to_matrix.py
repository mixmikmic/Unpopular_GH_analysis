import sys
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit/data_preprocessing_code')
sys.path.append('/Users/shayneufeld/GitHub/mouse_bandit')
import support_functions as sf
import numpy as np
import pandas as pd
import scipy as sp
import bandit_preprocessing as bp
import os
get_ipython().magic('matplotlib inline')

'''
load in trial data
'''
columns = ['Port Poked','Right Reward Prob','Left Reward Prob','Reward Given']

trial_df = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/trials_hmm_full_7030_greedy.csv',names=columns)

trial_df['Since last trial (s)'] = 0
trial_df['Trial Duration (s)'] = 0

trial_df.head(2)

feature_matrix = bp.create_reduced_feature_matrix(trial_df,'hmm_7030_greedy','02272017_2')

feature_matrix.head(5)

feature_matrix.to_csv('hmm_matrix_full_7030_greedy.csv')

