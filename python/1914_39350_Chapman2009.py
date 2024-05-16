import warnings
warnings.filterwarnings("ignore")

import pandas as pd

tbl1 = pd.read_csv("http://iopscience.iop.org/0004-637X/690/1/496/suppdata/apj291883t3_ascii.txt",
                   skiprows=[0,1,2,4], sep='\t', header=0, na_values=' ... ', skipfooter=6)
del tbl1["Unnamed: 6"]
tbl1

