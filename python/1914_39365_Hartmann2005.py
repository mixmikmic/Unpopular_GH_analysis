import warnings
warnings.filterwarnings("ignore")

import pandas as pd

names = ["Name","2MASS ID","R.A.(J2000.0)(deg)","Decl. (J2000.0)(deg)",
         "J(mag)","H(mag)","Ks(mag)","[3.6](mag)","[4.5](mag)",
         "[5.8](mag)","[8](mag)","JD-53000","IRAC Type"]
tbl1 = pd.read_csv("http://iopscience.iop.org/0004-637X/629/2/881/fulltext/61849.tb1.txt", 
                   na_values='\ldots',sep='\t', names=names)
tbl1.head()

