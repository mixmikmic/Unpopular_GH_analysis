import numpy as np
import scipy as sp
import math
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append("../../")
import pyApproxTools as pat
importlib.reload(pat)

get_ipython().magic('matplotlib inline')

fem_div = 7
field_div = 2

n = 20

widths = [2**i for i in range(fem_div-3)][::-1]

# Import data
bs_cs = []
bs_wcs = []


Wms_c = []
Wms_wc = []
for width in widths:
    Wms_c.append(pat.PWBasis(file_name='../../scripts/omp/Wm_c_{0}.npz'.format(width)))
    Wms_wc.append(pat.PWBasis(file_name='../../scripts/omp/Wm_wc_{0}.npz'.format(width)))
    
    bs_cs.append(np.load('../../scripts/omp/bs_c_{0}.npy'.format(width)))
    bs_wcs.append(np.load('../../scripts/omp/bs_wc_{0}.npy'.format(width)))
m = bs_cs[0].shape[0]

# Lets plot our measurment locations

fig = plt.figure(figsize=(15, 8))

for i, width in enumerate(widths):
    meas_c = Wms_c[i].vecs[0]
    for vec in Wms_c[i].vecs[1:]:
        meas_c += vec

    meas_wc = Wms_wc[i].vecs[0]
    for vec in Wms_wc[i].vecs[1:]:
        meas_wc += vec
    print('max ' + str(width) + ' Coll: ' + str(meas_c.values.max()) + ' WC: ' + str(meas_wc.values.max()))
    ax1 = fig.add_subplot(2, len(widths), i+1, projection='3d')
    meas_c.plot(ax1, title='Collective OMP, width {0}'.format(width))
    ax2 = fig.add_subplot(2, len(widths), i+1+len(widths), projection='3d')
    meas_wc.plot(ax2, title='Worst-case OMP, width {0}'.format(width))

sns.set_style('whitegrid')
line_style = ['-', '--', ':', '-', '-.']
pals = [ 'Blues_r', 'Reds_r', 'Greens_r', 'Purples_r']

bl = (51/255, 133/255, 255/255)
re = (255/255, 102/255, 102/255)

axs = []

fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for varying local avg widths')#, title=r'$\beta(V_n, W_m)$ against $m$ for various $n$')
    
sns.set_palette(pals[1])
cp = sns.color_palette()
for i, width in enumerate(widths[1:]):    
    plt.plot(range(m), bs_wcs[i], line_style[i], label=r'Worst-Case OMP $W_m$ for $\varepsilon={{{0}}} \times 2^{{{1}}}$'.format(width, -fem_div), color=re)#cp[i])

sns.set_palette(pals[0])
cp = sns.color_palette()
for i, width in enumerate(widths[1:]):
    plt.plot(range(m), bs_cs[i], line_style[i], label=r'Collective OMP $W_m$ for $\varepsilon={{{0}}} \times 2^{{{1}}}$'.format(width, -fem_div), color=bl)#cp[i])

ax.set(xlabel='m', ylabel=r'$\beta(V_n, W_m)$', xlim=[0,200], ylim=[0,1])#r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=4)
plt.savefig('2dCOMPvsWCOMPLocAvg.pdf')
plt.show()

v = Wm_wc.vecs[0]
for w in Wm_wc.vecs[1:]:
    v += w

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
v.plot(ax1)
plt.show()

fig = plt.figure(figsize=(8, 8))
ax2 = fig.add_subplot(1, 1, 1, projection='3d')
Vn.vecs[2].plot(ax2)
plt.show()



