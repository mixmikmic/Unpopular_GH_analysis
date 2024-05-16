import pysal as ps
import pandas as pd
import numpy as np

data = ps.pdio.read_files(ps.examples.get_path('NAT.shp'))
W = ps.queen_from_shapefile(ps.examples.get_path('NAT.shp'))
W.transform = 'r'

data.head()

I_HR90 = ps.Moran(data.HR90.values, W)

I_HR90.I, I_HR90.p_sim

I_HR90.sim[0:5]

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

sns.kdeplot(I_HR90.sim, shade=True)
plt.vlines(I_HR90.sim, 0, 1)
plt.vlines(I_HR90.I, 0, 40, 'r')

sns.kdeplot(I_HR90.sim, shade=True)
plt.vlines(I_HR90.sim, 0, 1)
plt.vlines(I_HR90.EI+.01, 0, 40, 'r')

c_HR90 = ps.Geary(data.HR90.values, W)
#ps.Gamma
#ps.Join_Counts

c_HR90.C, c_HR90.p_sim

bv_HRBLK = ps.Moran_BV(data.HR90.values, data.BLK90.values, W)

bv_HRBLK.I, bv_HRBLK.p_sim

LMo_HR90 = ps.Moran_Local(data.HR90.values, W)

LMo_HR90.Is, LMo_HR90.p_sim

LMo_HR90 = ps.Moran_Local(data.HR90.values, W, permutations=9999)

Lag_HR90 = ps.lag_spatial(W, data.HR90.values)
HR90 = data.HR90.values

sigs = HR90[LMo_HR90.p_sim <= .001]
W_sigs = Lag_HR90[LMo_HR90.p_sim <= .001]
insigs = HR90[LMo_HR90.p_sim > .001]
W_insigs = Lag_HR90[LMo_HR90.p_sim > .001]

b,a = np.polyfit(HR90, Lag_HR90, 1)

plt.plot(sigs, W_sigs, '.', color='firebrick')
plt.plot(insigs, W_insigs, '.k', alpha=.2)
 # dashed vert at mean of the last year's PCI
plt.vlines(HR90.mean(), Lag_HR90.min(), Lag_HR90.max(), linestyle='--')
 # dashed horizontal at mean of lagged PCI
plt.hlines(Lag_HR90.mean(), HR90.min(), HR90.max(), linestyle='--')

# red line of best fit using global I as slope
plt.plot(HR90, a + b*HR90, 'r')
plt.text(s='$I = %.3f$' % I_HR90.I, x=50, y=15, fontsize=18)
plt.title('Moran Scatterplot')
plt.ylabel('Spatial Lag of HR90')
plt.xlabel('HR90')

pd.get_dummies(data.SOUTH) #dummies for south (already binary)

pd.get_dummies(data.STATE_NAME) #dummies for state by name

y = data[['HR90']].values
x = data[['BLK90']].values

unrelated_effect = np.random.normal(0,100, size=y.shape[0]).reshape(y.shape)

X = np.hstack((x, unrelated_effect))

regimes = data.SOUTH.values.tolist()

regime_reg = ps.spreg.OLS_Regimes(y, X, regimes)

betas = regime_reg.betas
sebetas = np.sqrt(regime_reg.vm.diagonal())

sebetas

plt.plot(betas,'ok')
plt.axis([-1,6,-1,8])
plt.hlines(0,-1,6, color='k', linestyle='--')
plt.errorbar([0,1,2,3,4,5], betas.flatten(), yerr=sebetas*3, fmt='o', ecolor='r')
plt.xticks([-.5,0.5,1.5,2.5,3.5,4.5, 5.5], ['',
                                       'Not South: Constant',
                                       'Not South: BLK90',
                                       'Not South: Not South',
                                       'South: Constant',
                                       'South: South',
                                       'South: Unrelated',''], rotation='325')
plt.title('Regime Fixed Effects')

