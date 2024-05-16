import numpy as np
import pandas as pd

df = pd.read_csv('data/ufc_dot_com_fighter_data_CLEAN_28Feb2017.csv', header=0)
df.head(3)

df.info()

df = df[df.Active == 1][['Name', 'Record', 'Age', 'Weight', 'Height', 'Reach', 'LegReach']].reset_index(drop=True)
df.head(3)

rev = df.sort_index(ascending=False).reset_index(drop=True)
rev.head(3)

cmb = pd.merge(df, rev, how='inner', left_index=True, right_index=True)
cmb.Record_x = cmb.Record_x.str.replace('-', '--')
cmb.Record_y = cmb.Record_y.str.replace('-', '--')
cmb = cmb.fillna(0)
cmb = cmb.astype({'Age_x':int, 'Weight_x':int, 'Height_x':int, 'Reach_x':int, 'LegReach_x':int})
cmb = cmb.astype({'Age_y':int, 'Weight_y':int, 'Height_y':int, 'Reach_y':int, 'LegReach_y':int})
cmb.columns = ['Name', 'Record', 'Age', 'Wt.', 'Ht.', 'Rh.', 'Lg.', 'Name', 'Record', 'Age', 'Wt.','Ht.', 'Rh.', 'Lg.']
cmb.head(3)

import math
thres = math.ceil(df.shape[0] / 2.0)
iofile = 'ufc_table.tex'
cmb[:int(thres)].to_latex(iofile, index=False, na_rep='', longtable=True)

with open(iofile) as f:
     lines = f.readlines()
with open(iofile, 'w') as f:
     for line in lines:
          line = line.replace('llrrrrrllrrrrr','llrrrrr||llrrrrr')
          line = line.replace('\multicolumn{3}{r}','\multicolumn{14}{c}')
          line = line.replace('Continued on next page', 'Active UFC Fighters --- J. D. Halverson --- 2--28--2017')
          f.write(line)

