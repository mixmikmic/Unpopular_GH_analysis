import pandas as pd
import numpy as np
import os

os.chdir('/home/domestic-ra/Working/CPS_ORG/EPOPs/')
#os.chdir('C:/Working/econ_data/micro/')

# Python numbering (subtract one from first number in range)
colspecs = [(15,17), (17,21), (121,123), (128,130), (136,138), (392,394), (396,398), (526, 534), (602,612), (845,855)]
colnames = ['month', 'year', 'age', 'PESEX', 'PEEDUCA', 'PREMPNOT', 'PRFTLF', 'PRERNWA', 'orgwgt', 'fnlwgt']

educ_dict = {31: 'LTHS',
             32: 'LTHS',
             33: 'LTHS',
             34: 'LTHS',
             35: 'LTHS',
             36: 'LTHS',
             37: 'LTHS',
             38: 'HS',
             39: 'HS',
             40: 'Some college',
             41: 'Some college',
             42: 'Some college',
             43: 'College',
             44: 'Advanced',
             45: 'Advanced',
             46: 'Advanced',
            }

gender_dict = {1: 0, 2: 1}

empl_dict = {1: 1, 2: 0, 3: 0, 4: 0}

data = pd.DataFrame()   # This will be the combined annual df

for file in os.listdir('Data/'):
    if file.endswith('.dat'):
        df = pd.read_fwf('Data/{}'.format(file), colspecs=colspecs, header=None)
        # Set the values to match with CEPR extracts
        df.columns = colnames
        # Add the currently open monthly df to the combined annual df
        data = data.append(df)

data['educ'] = data['PEEDUCA'].map(educ_dict)
data['female'] = data['PESEX'].map(gender_dict)
data['empl'] = data['PREMPNOT'].map(empl_dict)
data['weekpay'] = data['PRERNWA'].astype(float) / 100
data['uhourse'] = data['PRFTLF'].replace(1, 40)

data.dropna().to_stata('Data/cepr_org_2017.dta')

data = data[data['year'] == 2017].dropna()
for month in sorted(data['month'].unique()):
    df = data[(data['female'] == 1) & 
              (data['age'].isin(range(25,55))) & # python equiv to 25-54
              (data['month'] == month)].dropna()
    # EPOP as numpy weighted average of the employed variable
    epop = np.average(df['empl'].astype(float), weights=df['fnlwgt']) * 100
    date = pd.to_datetime('{}-{}-01'.format(df['year'].values[0], month))
    print('{:%B %Y}: Women, age 25-54: {:0.1f}'.format(date, epop))

import wquantiles
df = data[(data['PRERNWA'] > -1) & 
          (data['age'] >= 16) & 
          (data['PRFTLF'] == 1) &
          (data['month'].isin([1, 2, 3]))].dropna()
print('2017 Q1 Usual Weekly Earnings: ${0:,.2f}'.format(
    # Weighted median using wquantiles package
    wquantiles.median(df['PRERNWA'], df['orgwgt']) / 100.0))

