import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

from helper_methods import read_json
df_tip = read_json('data/yelp_academic_dataset_tip.json')
df_tip.head()

df_tip.info()

df_tip.describe()

df_tip['num_characters'] = df_tip['text'].apply(len)
plt.plot(df_tip['date'], df_tip['num_characters'], '.')
plt.xlabel('Date')
plt.ylabel('Number of characters in text')

most_chars = df_tip.sort_values('num_characters', ascending=False)
most_chars.loc[most_chars.index[0], 'text']

most_chars.loc[most_chars.index[1], 'text']

most_chars.loc[most_chars.index[2], 'text']

plt.hist(df_tip.num_characters, bins=40)
plt.xlabel('Number of characters')
plt.ylabel('Count')

