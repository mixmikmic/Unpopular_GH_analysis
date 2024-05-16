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

data = pd.read_csv('/Users/shayneufeld/GitHub/mouse_bandit/data/processed_data/new_master_8020_df.csv',index_col=0)

data.head(5)

data['Session ID'][0]

def encode_categorical(array):
    if not (array.dtype == np.dtype('float64') or array.dtype == np.dtype('int64')) :
        return preprocessing.LabelEncoder().fit_transform(array) 
    else:
        return array

categorical = (data.dtypes.values != np.dtype('float64'))
data_1hot = data.apply(encode_categorical)

# Apply one hot endcoing
encoder = preprocessing.OneHotEncoder(categorical_features=categorical, sparse=False)  # Last value in mask is y
x = encoder.fit_transform(data_1hot.values)

encoder.feature_indices_

encoder.active_features_

np.unique(data['Session ID'].values)

x.shape

data_encoded = bp.OneHotEncode(data)

master_matrix.to_csv(os.path.join(root_dir,'new_master_8020_df.csv'))

master_matrix

