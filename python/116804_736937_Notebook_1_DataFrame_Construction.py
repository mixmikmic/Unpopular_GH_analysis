import pandas as pd
import arrow # way better than datetime
import numpy as np
import random
import re
get_ipython().magic('run helper_functions.py')

df = pd.read_csv("tweets_formatted.txt", sep="| |", header=None)

df.shape

list_of_dicts = []
for i in range(df.shape[0]):
    temp_dict = {}
    temp_lst = df.iloc[i,0].split("||")
    
    temp_dict['handle'] = temp_lst[0]
    temp_dict['tweet'] = temp_lst[1]
    
    try: #sometimes the date/time is missing - we will have to infer
        temp_dict['date'] = arrow.get(temp_lst[2]).date()
    except:
        temp_dict['date'] = np.nan
    try:  
        temp_dict['time'] = arrow.get(temp_lst[2]).time()
    except:
        temp_dict['time'] = np.nan
    
    list_of_dicts.append(temp_dict)
    
    

list_of_dicts[0].keys()

new_df = pd.DataFrame(list_of_dicts) #magic!

new_df.head() #unsorted!

new_df.sort_values(by=['date', 'time'], ascending=False, inplace=True)
new_df.reset_index(inplace=True)
del new_df['index']
pickle_object(new_df, "new_df")

new_df.head() #sorted first on date and then on time

sample_duplicate_indicies = []
for i in new_df.index:
    if "Multiplayer #Poker" in new_df.iloc[i, 3]:
        sample_duplicate_indicies.append(i)

new_df.iloc[sample_duplicate_indicies, :]



