get_ipython().magic('pylab inline')

import seaborn as sns
sns.set_context("notebook", font_scale=1.5)

import warnings
warnings.filterwarnings("ignore")

from astropy.io import ascii

tbl4 = ascii.read("http://iopscience.iop.org/0004-637X/794/1/36/suppdata/apj500669t4_mrt.txt")

tbl4[0:4]

Na_mask = ((tbl4["f_EWNaI"] == "Y") | (tbl4["f_EWNaI"] == "N"))
print "There are {} sources with Na I line detections out of {} sources in the catalog".format(Na_mask.sum(), len(tbl4))

tbl4_late = tbl4[['Name', '2MASS', 'SpType', 'e_SpType','EWHa', 'f_EWHa', 'EWNaI', 'e_EWNaI', 'f_EWNaI']][Na_mask]

tbl4_late.pprint(max_lines=100, )

