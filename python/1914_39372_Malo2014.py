import warnings
warnings.filterwarnings("ignore")

from astropy.io import ascii

import pandas as pd

#! mkdir ../data/Malo2014
#! wget http://iopscience.iop.org/0004-637X/788/1/81/suppdata/apj494919t7_mrt.txt

get_ipython().system(' head ../data/Malo2014/apj494919t7_mrt.txt')

from astropy.table import Table, Column

t1 = Table.read("../data/Malo2014/apj494919t7_mrt.txt", format='ascii')  

sns.distplot(t1['Jmag'].data.data)

t1



