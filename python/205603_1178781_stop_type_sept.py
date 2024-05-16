import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
get_ipython().magic('matplotlib inline')
pd.set_option('display.max_columns', None)
get_ipython().magic('config IPCompleter.greedy=True')

raw_veh_stop_sept = pd.read_csv('trimet_congestion/init_veh_stoph 1-30SEP2017.csv')

tripsh_sept = pd.read_csv('trimet_congestion/init_tripsh 1-30SEP2017.csv')

tripsh_sept.info(verbose=True, null_counts=True)

raw_veh_stop_sept.info(verbose=True, null_counts=True)

stop_type = tripsh_sept.merge(raw_veh_stop_sept, left_on='EVENT_NO', right_on='EVENT_NO_TRIP', how='inner')

stop_type.info(verbose=True, null_counts=True)

stop_type['TIME_DIFF'] = stop_type['ACT_DEP_TIME_y'] - stop_type['ACT_ARR_TIME']
stop_type.head()

stop_type = stop_type[stop_type.TIME_DIFF != 0]
stop_type.info(verbose=True, null_counts=True)

stop_type['TIME_DIFF_MIN'] = stop_type['TIME_DIFF'] / 60
stop_type.head()

stop_type[(stop_type['LINE_ID'] == 72)].groupby(['STOP_TYPE'])['TIME_DIFF'].sum().plot(title = 'Line 72 Stop Type in Seconds', kind='bar', y= 'seconds')

stop_type[(stop_type['LINE_ID'] == 33)].groupby(['STOP_TYPE'])['TIME_DIFF'].sum().plot(title = 'Line 33 Stop Type in Seconds', kind='bar', y= 'seconds')

stop_type[(stop_type['LINE_ID'] == 4)].groupby(['STOP_TYPE'])['TIME_DIFF'].sum().plot(title = 'Line 4 Stop Type in Seconds', kind='bar', y= 'seconds')

stop_type['LINE_ID'].value_counts()

stop_type[(stop_type['LINE_ID'] == 75)].groupby(['STOP_TYPE'])['TIME_DIFF'].sum().plot(title = 'Line 75 Stop Type in Seconds', kind='bar', y= 'seconds')

fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(12,12))
#from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})

stop_type[(stop_type['LINE_ID'] == 75)].groupby(['STOP_TYPE'])['TIME_DIFF_MIN'].sum().plot(ax=axes[0,0], kind='bar', y= 'seconds'); axes[0,0].set_title('Line 75 Stop Type in Seconds')
stop_type[(stop_type['LINE_ID'] == 4)].groupby(['STOP_TYPE'])['TIME_DIFF_MIN'].sum().plot(ax=axes[0,1], kind='bar', y= 'seconds'); axes[0,1].set_title('Line 4 Stop Type in Seconds')
stop_type[(stop_type['LINE_ID'] == 72)].groupby(['STOP_TYPE'])['TIME_DIFF_MIN'].sum().plot(ax=axes[1,0], kind='bar', y= 'seconds'); axes[1,0].set_title('Line 72 Stop Type in Seconds')
stop_type[(stop_type['LINE_ID'] == 20)].groupby(['STOP_TYPE'])['TIME_DIFF_MIN'].sum().plot(ax=axes[1,1], kind='bar', y= 'seconds'); axes[1,1].set_title('Line 20 Stop Type in Seconds')
#plt.tight_layout()

line4_df = stop_type[(stop_type.LINE_ID == 4) & (stop_type.STOP_TYPE == 3)]
line4_df.info(verbose=True, null_counts=True)

line4_df.head(100)

line14_df = stop_type[(stop_type.LINE_ID == 14) & (stop_type.STOP_TYPE == 3)]
line14_df.info(verbose=True, null_counts=True)

line73_df = stop_type[(stop_type.LINE_ID == 73) & (stop_type.STOP_TYPE == 3)]
line73_df.info(verbose=True, null_counts=True)

all_lines_disturbance_df = pd.concat([line4_df,line14_df,line73_df],ignore_index=True)
all_lines_disturbance_df.info(verbose=True, null_counts=True)

all_lines_disturbance_df.to_csv('Lines4_14_73_Disturbance_Stops.csv')



