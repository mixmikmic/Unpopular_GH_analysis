import warnings
warnings.filterwarnings("ignore")

from astropy.io import ascii

import pandas as pd

names = ["No.","R.A. (J2000)","Decl. (J2000)","i (mag)","J (mag)","K_s (mag)","T_eff (K)","A_V","Notes"]
tbl2 = pd.read_csv("http://iopscience.iop.org/0004-637X/726/1/23/suppdata/apj373191t2_ascii.txt",
                   sep="\t", skiprows=[0,1,2,3], na_values="sat", names = names)
tbl2.dropna(how="all", inplace=True)
tbl2.head()

get_ipython().system(' mkdir ../data/Geers2011')

tbl2.to_csv("../data/Geers2011/tb2.csv", index=False, sep='\t')

