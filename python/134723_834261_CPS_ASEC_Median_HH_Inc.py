import pandas as pd
import wquantiles

cols = ['HTOTVAL', 'H_HHTYPE', 'HSUP_WGT', 'H_SEQ', 'HEFAMINC']
df = pd.read_csv('data/2017_CPS_ASEC.csv', usecols=cols)
df = df[df['H_HHTYPE'] == 1]
df = df.drop_duplicates(subset='H_SEQ', keep='first')

'2016 Median HH Income: ${0:,.2f}'.format(wquantiles.median(df['HTOTVAL'], df['HSUP_WGT']))

df['HTOTVAL'].hist(bins=500, figsize=(15, 2))

