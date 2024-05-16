get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.options.display.max_columns = 150

import astropy

from astropy.io import fits
hdulist1 = fits.open('../data/Cottaar2014/per_epoch.fit')
hdulist2 = fits.open('../data/Cottaar2014/per_star.fit')

table1 = hdulist1[1]

table1.columns

table1.data.shape

table1.size

table1

from astropy.table import Table

tt = Table(data=table1.data)

tt.write('../data/Cottaar2014/per_epoch.csv', format='csv')

dat = pd.read_csv('../data/Cottaar2014/per_epoch.csv')

dat.head()

dat.columns

sns.set_context('talk', font_scale=1.0)

get_ipython().magic("config InlineBackend.figure_format = 'svg'")

dat.columns

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

plt.figure(figsize=[10, 6])
sc = plt.scatter(dat['Teff'], dat['log(g)'], c=dat['R_H'], vmin=0, vmax=2, s=35, cmap=cmap, alpha=0.5)
plt.colorbar(sc)
plt.xlabel('$T_{eff}$')
plt.ylabel('$\log{g}$')
plt.title('Cottaar et al. 2014 APOGEE/INSYNC data')
plt.xlim(7000, 2500)
plt.ylim(5.5, 2.7)

import numpy as np

plt.figure(figsize=[6, 4])
plt.hist(dat['R_H'], bins=np.arange(0, 3, 0.15));
plt.xlabel('$R-H$')
plt.ylabel('$N$')

Tfi = (dat.Teff > 4000) & (dat.Teff < 4200) 
lgi = (dat['log(g)'] > 3.5) & (dat['log(g)'] < 4.0)

gi = Tfi & lgi

dat.shape

gi.sum()

dat['2MASS'][gi]

table2 = hdulist2[1]

table2.columns

table2.data.shape

t2 = Table(data=table2.data)

t2.write('../data/Cottaar2014/per_star.csv', format='csv')

data = pd.read_csv('../data/Cottaar2014/per_star.csv')

data.head()

data.Cluster.unique()

data.columns

ic = data.Cluster == 'IC 348'
pl = data.Cluster == 'Pleiades'

bcah = pd.read_csv('../data/BCAH2002/BCAH2002_isochrones.csv', sep = '\t')
groups =bcah.groupby(by='Age')



plt.figure(figsize=[10, 6])
plt.scatter(data['Teff'][ic], data['log(g)'][ic], label='IC 348', c='r')
plt.scatter(data['Teff'][pl], data['log(g)'][pl], label='Pleiades')
plt.xlabel('$T_{eff}$')
plt.ylabel('$\log{g}$')
plt.title('Cottaar et al. 2014 APOGEE/INSYNC data')
plt.legend(loc='best')

for age, group in groups:
    no_decimal = np.abs(np.mod(age, 1)) <0.001
    if no_decimal:
        plt.plot(group.Teff, group.logg, 'k-', alpha=0.5, label='{:0.1f} Myr'.format(age))

plt.xlim(7000, 2500)
plt.ylim(5.5, 2.7)

data.columns

data.shape

sns.distplot(data['R_H'][ic], hist=False, label='IC 348')
sns.distplot(data['R_H'][pl], hist=False, label='Pleiades')



