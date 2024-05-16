# %install_ext http://raw.github.com/jrjohansson/version_information/master/version_information.py
get_ipython().magic('load_ext version_information')
get_ipython().magic('reload_ext version_information')
get_ipython().magic('version_information numpy, scipy, matplotlib, pandas')

import numpy as np
import pandas as pd

get_ipython().system('pwd')

# The cleaned data file is saved here:
output_file = "../data/coal_prod_cleaned.csv"

df1 = pd.read_csv("../data/coal_prod_2002.csv", index_col="MSHA_ID")
df2 = pd.read_csv("../data/coal_prod_2003.csv", index_col="MSHA_ID")
df3 = pd.read_csv("../data/coal_prod_2004.csv", index_col="MSHA_ID")
df4 = pd.read_csv("../data/coal_prod_2005.csv", index_col="MSHA_ID")
df5 = pd.read_csv("../data/coal_prod_2006.csv", index_col="MSHA_ID")
df6 = pd.read_csv("../data/coal_prod_2007.csv", index_col="MSHA_ID")
df7 = pd.read_csv("../data/coal_prod_2008.csv", index_col="MSHA_ID")
df8 = pd.read_csv("../data/coal_prod_2009.csv", index_col="MSHA_ID")
df9 = pd.read_csv("../data/coal_prod_2010.csv", index_col="MSHA_ID")
df10 = pd.read_csv("../data/coal_prod_2011.csv", index_col="MSHA_ID")
df11 = pd.read_csv("../data/coal_prod_2012.csv", index_col="MSHA_ID")

dframe = pd.concat((df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11))

# Noticed a probable typo in the data set: 
dframe['Company_Type'].unique()

# Correcting the Company_Type
dframe.loc[dframe['Company_Type'] == 'Indepedent Producer Operator', 'Company_Type'] = 'Independent Producer Operator'
dframe.head()

dframe[dframe.Year == 2003].head()

dframe.to_csv(output_file, )



