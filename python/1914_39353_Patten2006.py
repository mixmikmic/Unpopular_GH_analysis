get_ipython().magic('pylab inline')
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

def strip_parentheses(col, df):
    '''
    splits single column strings of "value (error)" into two columns of value and error
    
    input:
    -string name of column to split in two
    -dataframe to apply to
    
    returns dataframe
    '''
    
    out1 = df[col].str.replace(")","").str.split(pat="(")
    df_out = out1.apply(pd.Series)
    
    # Split the string on the whitespace 
    base, sufx =  col.split(" ")
    df[base] = df_out[0].copy()
    df[base+"_e"] = df_out[1].copy()
    del df[col]
    
    return df
    

names = ["Name","R.A. (J2000.0)","Decl. (J2000.0)","Spectral Type","SpectralType Ref.","Parallax (error)(arcsec)",
         "Parallax Ref.","J (error)","H (error)","Ks (error)","JHKRef.","PhotSys"]

tbl1 = pd.read_csv("http://iopscience.iop.org/0004-637X/651/1/502/fulltext/64991.tb1.txt", 
                   sep='\t', names=names, na_values='\ldots')

cols_to_fix = [col for col in tbl1.columns.values if "(error)" in col]
for col in cols_to_fix:
    print col
    tbl1 = strip_parentheses(col, tbl1)

tbl1.head()

names = ["Name","Spectral Type","[3.6] (error)","n1","[4.5] (error)","n2",
         "[5.8] (error)","n3","[8.0] (error)","n4","[3.6]-[4.5]","[4.5]-[5.8]","[5.8]-[8.0]","Notes"]

tbl3 = pd.read_csv("http://iopscience.iop.org/0004-637X/651/1/502/fulltext/64991.tb3.txt", 
                   sep='\t', names=names, na_values='\ldots')

cols_to_fix = [col for col in tbl3.columns.values if "(error)" in col]
cols_to_fix
for col in cols_to_fix:
    print col
    tbl3 = strip_parentheses(col, tbl3)

tbl3.head()

pd.options.display.max_columns = 50

del tbl3["Spectral Type"] #This is repeated

patten2006 = pd.merge(tbl1, tbl3, how="outer", on="Name")
patten2006.head()

import gully_custom

patten2006["SpT_num"], _1, _2, _3= gully_custom.specTypePlus(patten2006["Spectral Type"])

sns.set_context("notebook", font_scale=1.5)

for color in ["[3.6]-[4.5]", "[4.5]-[5.8]", "[5.8]-[8.0]"]:
    plt.plot(patten2006["SpT_num"], patten2006[color], '.', label=color)
    
plt.xlabel(r'Spectral Type (M0 = 0)')
plt.ylabel(r'$[3.6]-[4.5]$')
plt.title("IRAC colors as a function of spectral type")
plt.legend(loc='best')

patten2006.to_csv('../data/Patten2006/patten2006.csv', index=False)

