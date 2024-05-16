get_ipython().magic('matplotlib inline')

# Imports 
import csv 
import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

from itertools import groupby
from operator import itemgetter

# Load the series data 
info = pd.read_csv('../data/bls/series.csv')

def series_info(blsid, info=info):
    return info[info.blsid == blsid]

# Use this function to lookup specific BLS series info. 
series_info("LNS14000025")

# Load each series, grouping by BLS ID
def load_series_records(path='../data/bls/records.csv'):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        
        for blsid, rows in groupby(reader, itemgetter('blsid')):
            # Read all the data from the file and sort 
            rows = list(rows) 
            rows.sort(key=itemgetter('period'))
            
            # Extract specific data from each row, namely:
            # The period at the month granularity 
            # The value as a float 
            periods = [pd.Period(row['period']).asfreq('M') for row in rows]
            values = [float(row['value']) for row in rows]
            
            yield pd.Series(values, index=periods, name=blsid)
            

series = pd.concat(list(load_series_records()), axis=1)
series

