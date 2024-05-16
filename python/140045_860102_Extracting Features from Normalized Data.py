import dill
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')

data = dill.load(open('normalized_stock_price_slices.pkl', 'r'))

data[0][3]

from tsfresh import extract_features

testCase = data[10]

testCase

testFrame = pd.DataFrame(testCase[3][0])

testFrame['tick'] = "foo"

testFrame

testFrame.plot()

extracted_features = extract_features(testFrame, column_sort=0, column_id='tick')

extracted_features

from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

impute(extracted_features)
features_filtered = select_features(extracted_features, np.ndarray(int(testCase[2])))



