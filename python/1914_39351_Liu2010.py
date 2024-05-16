import warnings
warnings.filterwarnings("ignore")

from astropy.io import ascii

import pandas as pd

#! mkdir ../data/Liu2010
#! curl http://iopscience.iop.org/0004-637X/722/1/311/suppdata/apj336343t6_ascii.txt > ../data/Liu2010/apj336343t6_ascii.txt

tbl6 = pd.read_csv("../data/Liu2010/apj336343t6_ascii.txt",
                   sep="\t", na_values=" ... ", skiprows=[0,1,2], skipfooter=1, usecols=range(9))
tbl6.head()

J_s = tbl6.loc[0]

coeffs = J_s[["c_"+str(i) for i in range(6, -1, -1)]].values

func = np.poly1d(coeffs)

print(func)

