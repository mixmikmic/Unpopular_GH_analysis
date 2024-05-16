# Input the DB to Memory
import pandas as pd
import numpy as np
print("Loading DB...")
dfs = pd.read_csv("terrorism_red_cat_for_random_forest.csv")
print("DB Read...")
#print(data_file.sheet_names)
#dfs = data_file.parse(data_file.sheet_names[0])
#print("DB Parsed...")
del(dfs['Unnamed: 0'])

print(dfs.columns)

dimensions = dfs.columns.tolist()

columns = dfs.columns
for cols in columns:
    if cols == 'gname':
        continue
    if cols not in dimensions:
        del(dfs[cols])

columns = dfs.columns
print(columns)
print(dimensions)

del(dfs['gname'])

from sklearn.model_selection import train_test_split
df_train_test, df_val = train_test_split(dfs, test_size=0.5, random_state=42)

print(len(df_train_test))
print(len(df_val))

df_train_test.to_csv('terrorism_50_train_test.csv',encoding = 'utf-8')
df_val.to_csv('terrorism_50_val.csv',encoding = 'utf-8')



