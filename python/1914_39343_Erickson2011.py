get_ipython().magic('pylab inline')

import seaborn as sns
sns.set_context("notebook", font_scale=1.5)

#import warnings
#warnings.filterwarnings("ignore")

import pandas as pd

addr = "http://iopscience.iop.org/1538-3881/142/4/140/suppdata/aj403656t2_ascii.txt"
names = ['F', 'Ap', 'Alt_Names', 'X-Ray ID', 'RA', 'DEC', 'Li', 'EW_Ha', 'I', 'R-I',
       'SpT_Lit', 'Spectral_Type', 'Adopt', 'Notes', 'blank']
tbl2 = pd.read_csv(addr, sep='\t', skiprows=[0,1,2,3,4], skipfooter=7, engine='python', na_values=" ... ", 
                   index_col=False, names = names, usecols=range(len(names)-1))
tbl2.head()

addr = "http://iopscience.iop.org/1538-3881/142/4/140/suppdata/aj403656t3_ascii.txt"
names = ['F', 'Ap', 'Alt_Names', 'WMR', 'Spectral_Type', 'A_v', 'M_I',
       'log_T_eff', 'log_L_bol', 'Mass', 'log_age', 'Criteria', 'Notes', 'blank']
tbl3 = pd.read_csv(addr, sep='\t', skiprows=[0,1,2,3,4], skipfooter=9, engine='python', na_values=" ... ", 
                   index_col=False, names = names, usecols=range(len(names)-1))
tbl3.head()

get_ipython().system(' mkdir ../data/Erickson2011')

plt.plot(10**tbl3.log_T_eff, 10**tbl3.log_L_bol, '.')
plt.yscale("log")
plt.xlim(5000, 2000)
plt.ylim(1.0E-4, 1.0E1)
plt.xlabel(r"$T_{eff}$")
plt.ylabel(r"$L/L_{sun}$")
plt.title("Erickson et al. 2011 Table 3 HR Diagram")

tbl2.to_csv("../data/Erickson2011/tbl2.csv", sep="\t", index=False)
tbl3.to_csv("../data/Erickson2011/tbl3.csv", sep="\t", index=False)

