import warnings
warnings.filterwarnings("ignore")

from astropy.io import ascii

import pandas as pd

tbl2_vo = ascii.read("http://iopscience.iop.org/1538-3881/130/4/1733/fulltext/datafile2.txt")
tbl2_vo[0:3]

tbl4_vo = ascii.read("http://iopscience.iop.org/1538-3881/130/4/1733/fulltext/datafile4.txt")
tbl4_vo[0:4]

tbl2 = tbl2_vo.to_pandas()
del tbl2["Name"]
tbl2.rename(columns={'Note':"Flag"}, inplace=True)
tbl4 = tbl4_vo.to_pandas()

wilking2005 = pd.merge(tbl2, tbl4, how="right", on=["Field", "Aper"])

wilking2005

wilking2005["RA"] = wilking2005.RAh.astype(str) + wilking2005.RAm.astype(str) + wilking2005.RAs.astype(str)

wilking2005.RA

get_ipython().system(' mkdir ../data/Wilking2005')

wilking2005.to_csv("../data/Wilking2005/Wilking2005.csv", index=False, sep='\t')

