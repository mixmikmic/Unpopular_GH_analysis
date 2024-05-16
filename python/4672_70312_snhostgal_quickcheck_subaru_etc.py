get_ipython().magic('matplotlib inline')
get_ipython().magic('pdb')
#import time
#tstart = time.time()

import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt
import snhostspec
from glob import glob

reload(snhostspec)
sim2 = snhostspec.SnanaSimData()
sim2.load_simdata_catalog('wfirst_snhostspec2.cat')

idsubset = sim2.get_host_percentile_indices(zlist=[1.5,1.8,2.2])
sim2.simulate_subaru_snr_curves(np.ravel(idsubset))

fig = plt.figure(figsize=[13,3])
for et, iax  in zip([1, 5, 10,40], [1,2,3,4]):
    ax = fig.add_subplot(1,4,iax)
    ietwin = np.where((zcheck['exptime']==et) & (zcheck['gotz']>0))[0]
    ietfail = np.where((zcheck['exptime']==et) & (zcheck['gotz']<1))[0]
    ax.plot(zcheck['z'][ietwin], zcheck['mag'][ietwin], 'ro', ls=' ')
    ax.plot(zcheck['z'][ietfail], zcheck['mag'][ietfail], 'ks', ls=' ')
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)
ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
ax1.set_title('1hr')
ax2.set_title('5hr')
ax3.set_title('10hr')
ax1.set_ylabel('host gal H band mag')
ax2.set_xlabel('redshift')

mastercat = wfirst.WfirstMasterHostCatalog()
mastercat.load_all_simdata()
mastercat.write('wfirst_snhostspec_master.cat')

mastercat2 = wfirst.WfirstMasterHostCatalog()
mastercat2.read('wfirst_snhostspec_master.cat')

mastercat2.mastercat = ascii.read('wfirst_snhostspec_master.cat', format='commented_header')

mastercat2.simulate_all_seds()

reload(wfirst)
sim = wfirst.WfirstSimData('SNANA.SIM.OUTPUT/IMG_2T_4FILT_MD_SLT3_Z08_Ia-01_HEAD.FITS')
sim.load_matchdata('3DHST/3dhst_master.phot.v4.1.cat.FITS')
sim.get_matchlists()
sim.simdata.write("wfirst_snhostgal_sim.dat", format='ascii.commented_header')

reload(wfirst)

get_ipython().magic('pwd')

get_ipython().magic('pinfo sim.snanadata.read')

simlist = []
simfilelist_med = glob('SNANA.SIM.OUTPUT/*Z08*HEAD.FITS')
simfilelist_deep = glob('SNANA.SIM.OUTPUT/*Z17*HEAD.FITS')
hostz_med, hostmag_med = np.array([]), np.array([])
for simfile in simfilelist_med:
    sim = wfirst.WfirstSimData(simfile)
    sim.load_matchdata('3DHST/3dhst_master.phot.v4.1.cat.FITS')
    sim.get_matchlists()
    hostz_med = np.append(hostz_med, sim.zsim)
    hostmag_med = np.append(hostmag_med, sim.mag)
    simlist.append(sim)

hostz_deep, hostmag_deep = np.array([]), np.array([])
for simfile in simfilelist_deep:
    sim = wfirst.WfirstSimData(simfile)
    sim.load_matchdata('3DHST/3dhst_master.phot.v4.1.cat.FITS')
    sim.get_matchlists()
    hostz_deep = np.append(hostz_deep, sim.zsim)
    hostmag_deep = np.append(hostmag_deep, sim.mag)
    simlist.append(sim)    

if not os.path.isdir('3DHST/sedsim.output'):
    os.mkdir('3DHST/sedsim.output')
for sim in simlist:
    sim.load_sed_data()
    sim.simulate_seds()

# Example of a spectrum plot
eazyspecsim = wfirst.EazySpecSim('3DHST/sedsim.output/wfirst_simsed.AEGIS.0185.dat')
eazyspecsim.plot()

photdat3d = fits.open('3DHST/3dhst_master.phot.v4.1/3dhst_master.phot.v4.1.cat.FITS')
f160 = photdat3d[1].data['f_F160W']
zspec = photdat3d[1].data['z_spec']
zphot = photdat3d[1].data['z_peak']
zbest = np.where(zspec>0, zspec, zphot)
usephot = photdat3d[1].data['use_phot']

ivalid = np.where(((f160>0) & (zbest>0)) & (usephot==1) )[0]
mH3D = -2.5*np.log10(f160[ivalid])+25
z3D = zbest[ivalid]
plt.plot(z3D, mH3D, 'b.', ls=' ', ms=1, alpha=0.1)
#plt.plot(hostz_med, hostmag_med, 'g.', ls=' ', ms=3, alpha=0.3)
plt.plot(hostz_deep, hostmag_deep, 'r.', ls=' ', ms=3, alpha=0.3)
ax = plt.gca()
xlim = ax.set_xlim(0,2.5)
ylim = ax.set_ylim(28,20)
ax.set_xlabel('redshift')
ax.set_ylabel('host galaxy AB magnitude')

