import warnings
warnings.filterwarnings("ignore")

import pandas as pd

names = ["component", "RA", "Dec", "Spectral Type", "Teff", "AJ", "Lbol", "R-I","I", "J-H","H-Ks", "Ks", "Mass"]
tbl1 = pd.read_csv("http://iopscience.iop.org/0004-637X/614/1/398/fulltext/60660.tb1.txt", sep='\t', names=names)
tbl1

