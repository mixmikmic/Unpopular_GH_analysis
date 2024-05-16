get_ipython().magic('pylab inline')
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

tbl2 = pd.read_csv("http://iopscience.iop.org/0004-637X/703/1/399/suppdata/apj319072t2_ascii.txt", 
                   nrows=43, sep='\t', skiprows=2, na_values=[" sdotsdotsdot"])
tbl2.drop("Unnamed: 10",axis=1, inplace=True)

new_names = ['2MASS', 'Other_Names', 'Spectral_Type', 'T_eff', 'A_J','L_bol','Membership',
       'EW_Halpha', 'Basis of Selection', 'Night']
old_names = tbl2.columns.values
tbl2.rename(columns=dict(zip(old_names, new_names)), inplace=True)

tbl2.head()

sns.set_context("notebook", font_scale=1.5)

plt.plot(tbl2.T_eff, tbl2.L_bol, '.')
plt.ylabel(r"$L/L_{sun}$")
plt.xlabel(r"$T_{eff} (K)$")
plt.yscale("log")
plt.title("Luhman et al. 2009 Taurus Members")
plt.xlim(5000,2000)

get_ipython().system(' mkdir ../data/Luhman2009')

tbl2.to_csv("../data/Luhman2009/tbl2.csv", sep="\t")

