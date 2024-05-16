import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')

ess = pd.read_csv('ESSdata_Thinkful.csv')

ess.columns

ess.describe()

ess

ess_czch = ess.loc[
    ((ess['cntry'] == 'CZ') | (ess['cntry'] == 'CH')) & (ess['year'] == 6),
    ['cntry', 'tvtot', 'ppltrst', 'pplfair', 'pplhlp', 'happy', 'sclmeet']
]

ess_czch.head(10)

corr_matrix = ess_czch.corr()
print(corr_matrix)

f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_matrix, vmax=.8, square=True, cmap='YlGnBu')
plt.show()

# Restructure the data so we can use FacetGrid rather than making a boxplot
# for each variable separately.

df_long = ess_czch
df_long = pd.melt(ess_czch, id_vars=['cntry'])

df_long.head()

df_long.variable.unique()



