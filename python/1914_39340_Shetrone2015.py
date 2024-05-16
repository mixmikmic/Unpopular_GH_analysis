import pandas as pd

from astropy.io import ascii, votable, misc

#! mkdir ../data/Shetrone2015
#! wget http://iopscience.iop.org/0067-0049/221/2/24/suppdata/apjs521087t7_mrt.txt
#! mv apjs521087t7_mrt.txt ../data/Shetrone2015/
#! du -hs ../data/Shetrone2015/apjs521087t7_mrt.txt

dat = ascii.read('../data/Shetrone2015/apjs521087t7_mrt.txt')

get_ipython().system(' head ../data/Shetrone2015/apjs521087t7_mrt.txt')

dat.info

df = dat.to_pandas()

df.head()

df.columns

sns.distplot(df.Wave, norm_hist=False, kde=False)

df.count()

sns.lmplot('orggf', 'newgf', df, fit_reg=False)

from astropy import units as u

u.cm

EP1 = df.EP1.values*1.0/u.cm
EP2 = df.EP2.values*1.0/u.cm

EP1_eV = EP1.to(u.eV, equivalencies=u.equivalencies.spectral())
EP2_eV = EP2.to(u.eV, equivalencies=u.equivalencies.spectral())

deV = EP1_eV - EP2_eV

sns.distplot(deV)

plt.plot(df.Wave, deV, '.', alpha=0.05)
plt.xlabel('$\lambda (\AA)$')
plt.ylabel('$\Delta E \;(\mathrm{eV})$')

