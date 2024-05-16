import pandas as pd
import numpy as np

import requests
import json
from bs4 import BeautifulSoup

import dill
import re
import time

get_ipython().system(' head -n 10 ../priv/csv/cellartracker.txt')

with open('../priv/csv/cellartracker.txt','r') as fh:
    data_str = fh.read()

data_list = re.split(r"""\n\n""", data_str)

series_list = list()

for dat in data_list:
    
    dat_list = [x.strip() for x in dat.split('\n') 
                if (x.startswith('wine') or x.startswith('review'))]
    
    series_list.append(pd.Series(dict([re.search(r"""((?:wine|review)\/.+?): (.+)""", 
                                       x.strip()).groups() for x in dat_list])).T)
    

data_df = pd.concat(series_list, axis=1).T

data_df = data_df.rename_axis(lambda x: x.replace('/', '_'), axis=1)

data_df['wine_name'] = data_df.wine_name.apply(lambda x: x.replace('&#226;','a'))
data_df['review_text'] = data_df.review_text.apply(lambda x: x.replace('&#226;','a'))
# data_df['review_points'] = data_df.review_points.replace('N/A', np.NaN)
data_df = data_df.replace('N/A',np.NaN)

data_df.head()

data_df.isnull().sum()

data_df.to_pickle('../priv/pkl/03_cellartracker_dot_com_data.pkl')

