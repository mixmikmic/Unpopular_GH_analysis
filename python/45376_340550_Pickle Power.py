import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

get_ipython().magic('matplotlib inline')

# Feature data pickled here

# Power data pickled here
f = open('dataset/Electricity_P.csv')
totalPower = pd.read_csv(f,sep=',', header='infer', parse_dates=[1])
totalPower = totalPower.set_index('UNIX_TS')
totalPower.index = pd.to_datetime(totalPower.index)
powerWHE = totalPower
powerWHE['Power'] = totalPower['WHE']
powerWHE = pd.DataFrame(np.array(powerWHE['Power'],dtype=float),index=power.index, columns=['Power']).resample('5T').mean()

import pickle

pickle_file = open('dataset/Power.pkl','wb')
pickle.dump([powerWHE,totalPower],pickle_file)
pickle_file.close()



