import json
with open('Nail_Salon_Permits.json', 'r') as f:
    rawData = json.load(f)

rawData.keys()

rawData['data'][0]

len(rawData['meta']['view'])

column_names = []
types = []
for i in range(41):
    column = rawData['meta']['view']['columns'][i]['name']
    dtype = rawData['meta']['view']['columns'][i]['dataTypeName']
    column_names.append(column)
    types.append(dtype)
    print column, ' -- ', dtype

import pandas as pd
df = pd.DataFrame(rawData['data'], columns=column_names)

df.head(3).transpose()

df.dtypes

df.info()

neigh_counts = df['Salon Neighborhood'].value_counts()

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

fig, ax = plt.subplots(1, 1, figsize=(8,4))
plt.bar(range(len(neigh_counts)), neigh_counts, tick_label=neigh_counts.index)
plt.xticks(np.array(range(len(neigh_counts))) + 0.5, rotation=90)
plt.ylabel('Count')

df['Number Baths'].fillna(0, inplace=True)
df['Number Baths'] = df['Number Baths'].astype(int)

plt.hist(df['Number Baths'])
plt.xlabel('Number of Baths')
plt.ylabel('Count')

df['Number Tables'].isnull().value_counts()

df['Number Tables'].fillna(0, inplace=True)
df['Number Tables'] = df['Number Tables'].astype(int)

df.groupby('Salon Neighborhood').agg({'Number Tables': np.mean}).sort_values('Number Tables', ascending=False)

