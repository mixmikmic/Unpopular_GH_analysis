__depends__ = ['../outputs/llc_4320_kuroshio_pdfs.nc']
__dest__ = ['../writeup/figs/fig3.pdf']

import numpy as np

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from matplotlib.colors import LogNorm

from netCDF4 import Dataset

plt.rcParams['lines.linewidth'] = 2

pdfs  = Dataset(__depends__[0])

cpdf = np.logspace(-5,1.,7)

fig = plt.figure(figsize=(12,4))


ax = fig.add_subplot(241)
plt.contourf(pdfs['vorticity'][:],pdfs['strain'][:],pdfs['april/hourly']['pdf_vorticity_strain'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][100:],pdfs['vorticity'][100:],'k--')
plt.plot(pdfs['vorticity'][:100],-pdfs['vorticity'][:100],'k--')
plt.xlim(-4.,4.)
plt.ylim(0,4.)
plt.xticks([])
plt.yticks([])
plt.ylim(0.,4.)
plt.ylabel(r'Strain $\alpha/f$')
ticks=[0,2,4]
plt.yticks(ticks)
plt.text(-4.,4.15,'(a)',fontsize=14)
plt.title('Hourly',fontsize=11)

ax = fig.add_subplot(242)
plt.contourf(pdfs['vorticity'][:],pdfs['strain'][:],pdfs['april/daily-averaged']['pdf_vorticity_strain'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][100:],pdfs['vorticity'][100:],'k--')
plt.plot(pdfs['vorticity'][:100],-pdfs['vorticity'][:100],'k--')
plt.xlim(-4.,4.)
plt.ylim(0,4.)
plt.xticks([])
plt.yticks([])
plt.text(-4.,4.15,'(b)',fontsize=14)
plt.title('Daily-averaged',fontsize=11)

plt.text(-6,5.5,'April',fontsize=14)

ax = fig.add_subplot(2,4,3)
plt.contourf(pdfs['vorticity'][:],pdfs['strain'][:],pdfs['october/hourly']['pdf_vorticity_strain'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][100:],pdfs['vorticity'][100:],'k--')
plt.plot(pdfs['vorticity'][:100],-pdfs['vorticity'][:100],'k--')
plt.xlim(-4.,4.)
plt.ylim(0.,4.)
plt.yticks([])
xticks=[-4,-2,0,2,4]
plt.xticks([])
plt.text(-4.,4.15,'(c)',fontsize=14)
plt.title('Hourly',fontsize=11)

ax = fig.add_subplot(2,4,4)
plt.contourf(pdfs['vorticity'][:],pdfs['strain'][:],pdfs['october/daily-averaged']['pdf_vorticity_strain'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][100:],pdfs['vorticity'][100:],'k--')
plt.plot(pdfs['vorticity'][:100],-pdfs['vorticity'][:100],'k--')
plt.xlim(-4.,4.)
plt.ylim(0.,4.)
plt.xticks([])
plt.yticks([])
plt.text(-4.,4.15,'(d)',fontsize=14)
plt.title('Daily-averaged',fontsize=11)
plt.text(-6.75,5.5,'October',fontsize=14)

ax = fig.add_subplot(2,4,5)
plt.contourf(pdfs['vorticity'][:],pdfs['vorticity'][:],pdfs['april/hourly']['pdf_vorticity_lapssh'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][:],pdfs['vorticity'][:],'k--')
plt.xlim(-4.,4.)
plt.ylim(-4.,4.)
plt.ylabel(r'$(g/f^2) \, \nabla^2 \eta$')
ticks=[-4,-2,0,2,4]
plt.xticks(ticks)
plt.yticks(ticks)
plt.text(-4.,4.15,'(e)',fontsize=14)
#plt.xlabel(r'Vorticity $\zeta/f$')

ax = fig.add_subplot(2,4,6)
plt.contourf(pdfs['vorticity'][:],pdfs['vorticity'][:],pdfs['april/daily-averaged']['pdf_vorticity_lapssh'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][:],pdfs['vorticity'][:],'k--')
plt.xlim(-4.,4.)
plt.ylim(-4.,4.)
ticks=[-4,-2,0,2,4]
plt.xticks(ticks)
plt.yticks([])
plt.text(-4.,4.15,'(f)',fontsize=14)
#plt.xlabel(r'Vorticity $\zeta/f$')

ax = fig.add_subplot(2,4,7)
plt.contourf(pdfs['vorticity'][:],pdfs['vorticity'][:],pdfs['october/hourly']['pdf_vorticity_lapssh'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][:],pdfs['vorticity'][:],'k--')
plt.xlim(-4.,4.)
plt.ylim(-4.,4.)
plt.xticks(ticks)
ticks=[-4,-2,0,2,4]
plt.yticks([])
plt.text(-4.,4.15,'(g)',fontsize=14)
#plt.xlabel(r'Vorticity $\zeta/f$')

ax = fig.add_subplot(2,4,8)
cbs = plt.contourf(pdfs['vorticity'][:],pdfs['vorticity'][:],pdfs['october/daily-averaged']['pdf_vorticity_lapssh'][:],cpdf,vmin=1.e-5,vmax=10.,norm = LogNorm())
plt.plot(pdfs['vorticity'][:],pdfs['vorticity'][:],'k--')
plt.xlim(-4.,4.)
plt.ylim(-4.,4.)
plt.xticks(ticks)
plt.yticks([])
plt.text(-4.,4.15,'(h)',fontsize=14)
#plt.xlabel(r'Vorticity $\zeta/f$')


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.825, 0.16, 0.01, 0.7])
fig.colorbar(cbs, cax=cbar_ax,label=r'Probability density',extend='both',ticks=[1.e-5,1e-4,1.e-3,1e-2,1.e-1,1,10.])

plt.savefig(__dest__[0],dpi=150,bbox_inches='tight')





