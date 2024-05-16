import pandas as pd

messy = pd.read_csv('./data/weather-raw.csv')
messy

molten = pd.melt(messy,
                id_vars = ['id', 'year', 'month', 'element',],
                var_name = 'day');
molten.dropna(inplace = True)
molten = molten.reset_index(drop = True)
molten

def f(row):
    return "%d-%02d-%02d" % (row['year'], row['month'], int(row['day'][1:]))

molten['date'] = molten.apply(f, axis = 1)
molten = molten[['id', 'element', 'value', 'date']]
molten

tidy = molten.pivot(index='date', columns='element', values='value')
tidy

tidy = molten.groupby('id').apply(pd.DataFrame.pivot,
                                 index='date',
                                 columns='element',
                                 values='value')
tidy

tidy.reset_index(inplace=True)
tidy

import sys
import glob
import re

def extract_year(string):
    match = re.match(".+(\d{4})", string)
    if match != None: return match.group(1)
    
path = './data'
allFiles = glob.glob(path + "/201*-baby-names-illinois.csv")
frame = pd.DataFrame()
df_list = []
for file_ in allFiles:
    df = pd.read_csv(file_, index_col = None, header = 0)
    df.columns = map(str.lower, df.columns)
    df["year"] = extract_year(file_)
    df_list.append(df)
    
df = pd.concat(df_list)
df.head(10)

