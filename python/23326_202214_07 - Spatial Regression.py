import pysal as ps
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context('talk')
get_ipython().magic('matplotlib inline')

ps.examples.available()

ps.examples.explain('baltim')

data = ps.pdio.read_files(ps.examples.get_path('baltim'))

data.head()

mindist = ps.min_threshold_dist_from_shapefile(ps.examples.get_path('baltim.shp'))
mindist

W = ps.threshold_binaryW_from_array(np.array([data.X.values, data.Y.values]).T, 2*mindist)

W = ps.W(W.neighbors, W.weights)
W.transform = 'r'

ycols = ['PRICE']
xcols = ['NROOM', 'DWELL', 'LOTSZ', 'SQFT']#, 'AGE']#, 'NBATH', 'PATIO', 'FIREPL', 'AC', 'BMENT', 'NSTOR', 'GAR', ]
y = data[ycols].values
X = data[xcols].values

ols_reg = ps.spreg.OLS(y, X, w=W, spat_diag=True, moran=True, name_y=ycols, 
                       name_x = xcols)

print(ols_reg.summary)

effects, errs = ols_reg.betas, ols_reg.std_err

#plt.plot(range(0,len(effects.flatten())), effects.flatten(), '.k')
plt.title('Regression Effects plot')
plt.axis([-1,5, -12,30])
plt.errorbar(range(0,len(effects.flatten())), effects, yerr=errs.flatten()*2, fmt='.k', ecolor='r', capthick=True)
plt.hlines(0, -1, 5, linestyle='--', color='k')

resids = y - ols_reg.predy

Mresids = ps.Moran(resids.flatten(), W)

fig, ax = plt.subplots(1,3,figsize=(12*1.6,6))
for xi,yi,alpha in zip(data.X.values, data.Y.values, resids, ):
    if alpha+ ols_reg.std_y < 0:
        color='r'
    elif alpha - ols_reg.std_y > 0:
        color='b'
    else:
        color='k'
    ax[0].plot(xi,yi,color=color, marker='o', alpha = np.abs(alpha))#, alpha=alpha)
ax[0].axis([850, 1000, 500, 590])
ax[0].text(x=860, y=580, s='$I = %.3f (%.2f)$' % (Mresids.I, Mresids.p_sim))


ax[1].plot(ols_reg.predy, resids, 'o')
ax[1].axis([15,110,-60,120])
ax[1].hlines(0,0,150, linestyle='--', color='k')
ax[1].set_xlabel('Prediction')
ax[1].set_ylabel('Residuals')

H = np.dot(X, np.linalg.inv(np.dot(X.T, X)))
H = np.dot(H, X.T)

lev = H.diagonal().reshape(-1,1)

ax[2].plot(lev, resids, '.k')
ax[2].hlines(0,0,.2,linestyle='--', color='k')
ax[2].set_xlabel('Leverage')
ax[2].set_ylabel('Residuals')
ax[2].legend(labels=['Residuals'])

ax[0].set_axis_bgcolor('white')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[0].set_title('Spatial Error in House Price Prediction')
ax[1].set_title('Residuals vs. Prediction')
ax[2].set_title('Residuals vs. Leverage')


plt.show()

ml_lag = ps.spreg.ML_Lag(y, X, w=W)#, name_y=ycols, name_x = xcols)
effects, errs = ml_lag.betas, ml_lag.std_err

print(ml_lag.summary)

plt.title('Regression Effects plot')
plt.axis([-1,5, -38,20])
plt.errorbar(range(0,len(effects.flatten())), effects, yerr=errs.flatten()*2, fmt='.k', ecolor='r', capthick=True)
plt.hlines(0, -1, 13, linestyle='--', color='k')

resids = y - ml_lag.predy
Mresids = ps.Moran(resids.flatten(), W)

fig, ax = plt.subplots(1,3,figsize=(12*1.6,6))
for xi,yi,alpha in zip(data.X.values, data.Y.values, resids):
    if alpha+ ols_reg.std_y < 0:
        color='r'
    elif alpha - ols_reg.std_y > 0:
        color='b'
    else:
        color='k'
    ax[0].plot(xi,yi,color=color, marker='o', alpha = np.abs(alpha))#, alpha=alpha)
ax[0].axis([850, 1000, 500, 590])
ax[0].text(x=860, y=580, s='$I = %.3f (%.2f)$' % (Mresids.I, Mresids.p_sim))



ax[1].plot(ols_reg.predy, resids, 'o')
ax[1].axis([15,110,-60,120])
ax[1].hlines(0,0,150, linestyle='--', color='k')
ax[1].set_xlabel('Prediction')
ax[1].set_ylabel('Residuals')

XtXi = np.linalg.inv(np.dot(X.T, X))
H = np.dot(X, XtXi)
H = np.dot(H, X.T)

lev = H.diagonal().reshape(-1,1)

ax[2].plot(lev, resids, '.k')
ax[2].hlines(0,0,.25,linestyle='--', color='k')
ax[2].set_xlabel('Tangental Leverage')
ax[2].set_ylabel('Residuals')
ax[2].axis([-.01,.2,-60,120])

ax[0].set_axis_bgcolor('white')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[0].set_title('Spatial Error in House Price Prediction')
ax[1].set_title('Residuals vs. Prediction')
ax[2].set_title('Residuals vs. Tangental Leverage')


plt.show()

xcols.append('AGE')

X = data[xcols].values

reg_ommit = ps.spreg.OLS(y,X, name_y = ycols, name_x = xcols)
effects, errs = reg_ommit.betas, reg_ommit.std_err
print(reg_ommit.summary)

#plt.plot(range(0,len(effects.flatten())), effects.flatten(), '.k')
plt.title('Regression Effects plot')
plt.axis([-1,6, -5,28])
plt.errorbar(range(0,len(effects.flatten())), effects, yerr=errs.flatten()*2, fmt='.k', ecolor='r', capthick=True)
plt.hlines(0, -1, 13, linestyle='--', color='k')

resids = y - reg_ommit.predy
Mresids = ps.Moran(resids.flatten(), W)

fig, ax = plt.subplots(1,3,figsize=(12*1.6,6))
for xi,yi,alpha in zip(data.X.values, data.Y.values, resids, ):
    if alpha+ ols_reg.std_y < 0:
        color='r'
    elif alpha - ols_reg.std_y > 0:
        color='b'
    else:
        color='k'
    ax[0].plot(xi,yi,color=color, marker='o', alpha = np.abs(alpha))#, alpha=alpha)
ax[0].axis([850, 1000, 500, 590])
ax[0].text(x=860, y=580, s='$I = %.3f (%.2f)$' % (Mresids.I, Mresids.p_sim))



ax[1].plot(ols_reg.predy, resids, 'o')
ax[1].axis([15,110,-60,120])
ax[1].hlines(0,0,150, linestyle='--', color='k')
ax[1].set_xlabel('Prediction')
ax[1].set_ylabel('Residuals')

H = np.dot(X, np.linalg.inv(np.dot(X.T, X)))
H = np.dot(H, X.T)

lev = H.diagonal().reshape(-1,1)

ax[2].plot(lev, resids, '.k')
ax[2].hlines(0,0,.25,linestyle='--', color='k')
ax[2].set_xlabel('Leverage')
ax[2].set_ylabel('Residuals')
ax[2].legend(labels=['Residuals'])

ax[0].set_axis_bgcolor('white')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[0].set_title('Spatial Error in House Price Prediction')
ax[1].set_title('Residuals vs. Prediction')
ax[2].set_title('Residuals vs. Leverage')


plt.show()

xcols.extend(['NBATH', 'PATIO', 'FIREPL', 'AC', 'BMENT', 'NSTOR', 'GAR', ])
X = data[xcols].values
reg_ommit = ps.spreg.OLS(y,X, name_y = ycols, name_x = xcols)
effects, errs = reg_ommit.betas, reg_ommit.std_err
resids = y - reg_ommit.predy
print(reg_ommit.summary)

plt.title('Regression Effects plot')
plt.axis([-1,13, -12,35])
plt.errorbar(range(0,len(effects.flatten())), effects, yerr=errs.flatten()*2, fmt='.k', ecolor='r', capthick=True)
plt.hlines(0, -1, 13, linestyle='--', color='k', linewidth=.9)

Mresids = ps.Moran(resids, W)

fig, ax = plt.subplots(1,3,figsize=(12*1.6,6))
for xi,yi,alpha in zip(data.X.values, data.Y.values, resids, ):
    if alpha+ ols_reg.std_y < 0:
        color='r'
    elif alpha - ols_reg.std_y > 0:
        color='b'
    else:
        color='k'
    ax[0].plot(xi,yi,color=color, marker='o', alpha = np.abs(alpha))#, alpha=alpha)
ax[0].axis([850, 1000, 500, 590])
ax[0].text(x=860, y=580, s='$I = %.3f (%.2f)$' % (Mresids.I, Mresids.p_sim))


ax[1].plot(ols_reg.predy, resids, 'o')
ax[1].axis([15,110,-60,120])
ax[1].hlines(0,0,150, linestyle='--', color='k')
ax[1].set_xlabel('Prediction')
ax[1].set_ylabel('Residuals')


H = np.dot(X, np.linalg.inv(np.dot(X.T, X)))
H = np.dot(H, X.T)

lev = H.diagonal().reshape(-1,1)

ax[2].plot(lev, resids, '.k')
ax[2].hlines(0,0,.25,linestyle='--', color='k')
ax[2].set_xlabel('Leverage')
ax[2].set_ylabel('Residuals')
ax[2].legend(labels=['Residuals'])

ax[0].set_axis_bgcolor('white')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[0].set_title('Spatial Error in House Price Prediction')
ax[1].set_title('Residuals vs. Prediction')
ax[2].set_title('Residuals vs. Leverage')


plt.show()

reg_ommit = ps.spreg.ML_Lag(y,X, w=W)
effects, errs = reg_ommit.betas, reg_ommit.std_err
resids = y - reg_ommit.predy
print(reg_ommit.summary)

#plt.plot(range(0,len(effects.flatten())), effects.flatten(), '.k')
plt.title('Regression Effects plot')
plt.axis([-1,14, -10,20])
plt.errorbar(range(0,len(effects.flatten())), effects, yerr=errs.flatten(), fmt='.k', ecolor='r', capthick=True)
plt.hlines(0, -1, 14, linestyle='--', color='k')

Mresids = ps.Moran(resids, W)

fig, ax = plt.subplots(1,3,figsize=(12*1.6,6))
for xi,yi,alpha in zip(data.X.values, data.Y.values, resids, ):
    if alpha+ ols_reg.std_y < 0:
        color='r'
    elif alpha - ols_reg.std_y > 0:
        color='b'
    else:
        color='k'
    ax[0].plot(xi,yi,color=color, marker='o', alpha = np.abs(alpha))#, alpha=alpha)
ax[0].axis([850, 1000, 500, 590])
ax[0].text(x=860, y=580, s='$I = %.3f (%.2f)$' % (Mresids.I, Mresids.p_sim))


ax[1].plot(ols_reg.predy, resids, 'o')
ax[1].axis([15,110,-60,120])
ax[1].hlines(0,0,150, linestyle='--', color='k')
ax[1].set_xlabel('Prediction')
ax[1].set_ylabel('Residuals')


H = np.dot(X, np.linalg.inv(np.dot(X.T, X)))
H = np.dot(H, X.T)

lev = H.diagonal().reshape(-1,1)

ax[2].plot(lev, resids, '.k')
ax[2].hlines(0,0,.25,linestyle='--', color='k')
ax[2].set_xlabel('Tangental Leverage')
ax[2].set_ylabel('Residuals')

ax[0].set_axis_bgcolor('white')
ax[0].set_xticks([])
ax[0].set_yticks([])

ax[0].set_title('Spatial Error in House Price Prediction')
ax[1].set_title('Residuals vs. Prediction')
ax[2].set_title('Residuals vs. Leverage')


plt.show()



