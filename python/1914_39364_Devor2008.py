import pandas as pd

from astropy.io import ascii, votable, misc

#! mkdir ../data/Devor2008

#! curl http://iopscience.iop.org/1538-3881/135/3/850/suppdata/aj259648_mrt7.txt >> ../data/Devor2008/aj259648_mrt7.txt

get_ipython().system(' du -hs ../data/Devor2008/aj259648_mrt7.txt')

dat = ascii.read('../data/Devor2008/aj259648_mrt7.txt')

get_ipython().system(' head ../data/Devor2008/aj259648_mrt7.txt')

dat.info

df = dat.to_pandas()

df.head()

df.columns

sns.distplot(df.Per, norm_hist=False, kde=False)

gi = (df.RAh == 4) & (df.RAm == 16) & (df.DEd == 28) & (df.DEm == 7)

gi.sum()

df[gi].T

get_ipython().system(' head ../data/Devor2008/T-Tau0-01262.lc')

cols = ['HJD-2400000', 'r_band', 'r_unc']
lc_raw = pd.read_csv('../data/Devor2008/T-Tau0-01262.lc', names=cols, delim_whitespace=True)

lc_raw.head()

lc_raw.count()

sns.set_context('talk')

plt.plot(lc_raw['HJD-2400000'], lc_raw.r_band, '.')
plt.ylim(0.6, -0.6)

plt.plot(np.mod(lc_raw['HJD-2400000'], 3.375)/3.375, lc_raw.r_band, '.', alpha=0.5)
plt.xlabel('phase')
plt.ylabel('$\Delta \;\; r$')
plt.ylim(0.6, -0.6)

plt.plot(np.mod(lc_raw['HJD-2400000'], 6.74215), lc_raw.r_band, '.')
plt.ylim(0.6, -0.6)

get_ipython().system(' ls /Users/gully/Downloads/catalog/T-Tau0-* | head -n 10')

lc2 = pd.read_csv('/Users/gully/Downloads/catalog/T-Tau0-00397.lc', names=cols, delim_whitespace=True)
plt.plot(lc2['HJD-2400000'], lc2.r_band, '.')
plt.ylim(0.6, -0.6)

this_p = df.Per[df.Name == 'T-Tau0-00397']
plt.plot(np.mod(lc2['HJD-2400000'], this_p), lc2.r_band, '.', alpha=0.5)
plt.xlabel('phase')
plt.ylabel('$\Delta \;\; r$')
plt.ylim(0.6, -0.6)



