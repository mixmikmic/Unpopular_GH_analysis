# The normal imports
import numpy as np
from numpy.random import randn
import pandas as pd

# Import the stats librayr from numpy
from scipy import stats

# These are the plotting modules adn libraries we'll use:
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Command so that plots appear in the iPython Notebook
get_ipython().magic('matplotlib inline')

salary_df = pd.read_csv('Salary.csv')

salary_df.head()

position_lvl

position_new

position_lvl = pd.Series()
position_new = pd.Series()
lvl_dict = {"I " : 1, "II ": 2, "III " : 3, "IV ": 4, "V ": 5, "VI ": 6, "VII ": 7}
def get_position_lvl(sal_df):
    for idx,pos in sal_df.iterrows():
        new_pos = pos[1]
        temp = 0
        if pos[1][::-1][:2] in lvl_dict:
            temp = lvl_dict[pos[1][::-1][:2]]
            new_pos = new_pos[0:len(new_pos) - 2]
        elif pos[1][::-1][:3] in lvl_dict:
            temp = lvl_dict[pos[1][::-1][:3]]
            new_pos = new_pos[0:len(new_pos) - 3]
        elif pos[1][::-1][:4] in lvl_dict:
            temp = lvl_dict[pos[1][::-1][:4]]
            new_pos = new_pos[0:len(new_pos) - 4]        
        position_new.set_value(idx,new_pos )
        position_lvl.set_value(idx,temp )

salary_df[[1,2,4]].head()

get_position_lvl(salary_df[[1,2,4]])
salary_df.head()

salary_df['Position Title New'] =  position_new
salary_df['Position Level'] = position_lvl
salary_df.head(100)

pd.pivot_table(salary_df, values = 'YTD Gross Pay', index = ['Position Title New','Position Level'], columns=['Agency Name'], aggfunc=np.mean)

pd.pivot_table(salary_df, values = 'YTD Gross Pay', index = ['Position Title New','Position Level'], columns=['Agency Name'], aggfunc=np.std).dropna(how='all')

salary_df['YTD Gross Pay Range'] = pd.cut(salary_df['YTD Gross Pay'],500,precision=1)

salary_df

sorted(salary_df['YTD Gross Pay'], reverse=True)

plt.hist(salary_df['YTD Gross Pay'],bins=100)

sns.jointplot(salary_df['YTD Gross Pay'],salary_df['Position Level'])



