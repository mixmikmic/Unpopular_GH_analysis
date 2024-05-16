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

n = 50

np.random.seed(3)

Vn_sin = pat.make_pw_sin_basis(div=fem_div)
Vn_red, fields = pat.make_pw_reduced_basis(n, field_div=field_div, fem_div=fem_div)
Vn_red_o = Vn_red.orthonormalise()

# Lets plot our measurment locations
fig = plt.figure(figsize=(15, 6))

disp = 6
for i in range(disp):
    ax1 = fig.add_subplot(2, disp, i+1, projection='3d')
    Vn_sin.vecs[i].plot(ax1, title=r'sin $\omega_{{{0}}}$'.format(i+1))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.set(xlabel='', ylabel='')
    ax2 = fig.add_subplot(2, disp, i+1+disp, projection='3d')
    Vn_red_o.vecs[i].plot(ax2, title=r'reduced $\omega_{{{0}}}$'.format(i+1))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    ax2.set(xlabel='', ylabel='')
plt.show()

fig = plt.figure(figsize=(15,10))
disp = 12
for i in range(disp):
    ax1 = fig.add_subplot(3, disp/3, i+1, projection='3d')
    Vn_red_o.vecs[i].plot(ax1, title=r'$\phi_{{{0}}}$'.format(i+1))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.set(xlabel='', ylabel='')
plt.savefig('Vn_RedBasisOrtho.pdf')
plt.show()

num_sol = 10
sols, sol_fields = pat.make_pw_reduced_basis(num_sol, field_div=field_div, fem_div=fem_div)
soln_col = sols.vecs

dist_sin = np.zeros((num_sol, n))
dist_red = np.zeros((num_sol, n))

for i, v in enumerate(soln_col):
    for j in range(1,n):
        P_v_sin = Vn_sin.subspace(slice(0,j)).project(v)
        P_v_red = Vn_red.subspace(slice(0,j)).project(v)
        
        dist_sin[i, j] = (v - P_v_sin).norm()
        dist_red[i, j] = (v - P_v_red).norm()

width = 2

n = 20
m=150

print('Construct dictionary of local averages...')
D = pat.make_pw_hat_rep_dict(fem_div, width=width)

print('Worst-case greedy basis construction...')

wcbc = pat.WorstCaseOMP(D, Vn_sin.subspace(slice(0,20)), Wm=pat.PWBasis(), verbose=True)
Wm_wc_sin = wcbc.construct_to_m(m)
Wm_wc_sin_o = Wm_wc_sin.orthonormalise()

wcbc = pat.WorstCaseOMP(D, Vn_red_o.subspace(slice(0,20)), Wm=pat.PWBasis(), verbose=True)
Wm_wc_red = wcbc.construct_to_m(m)
Wm_wc_red_o = Wm_wc_red.orthonormalise()

bs_wc_sin = np.zeros(m)
bs_wc_red = np.zeros(m)

# For efficiency it makes sense to compute the basis pair and the associated
# cross-gramian only once, then sub sample it as we grow m...
BP_wc_sin_l = pat.BasisPair(Wm_wc_sin_o, Vn_sin.subspace(slice(0,20)))
BP_wc_red_l = pat.BasisPair(Wm_wc_red_o, Vn_red_o.subspace(slice(0,20)))

for i in range(n, m):
    BP_wc_sin =  BP_wc_sin_l.subspace(Wm_indices=slice(0,i))
    BP_wc_red =  BP_wc_red_l.subspace(Wm_indices=slice(0,i))

    bs_wc_sin[i] = BP_wc_sin.beta()
    bs_wc_red[i] = BP_wc_red.beta()

sns.set_style('whitegrid')
line_style = ['-', '--', ':', '-', '-.']

sns.set_palette("hls", 8)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for sinusoid and reduced bases with $n=20$')#, title=r'$\beta(V_n, W_m)$ against $m$ for various $n$')
    
plt.plot(range(m), bs_wc_sin, label=r'Sinusoid basis')
plt.plot(range(m), bs_wc_red, label=r'Reduced basis')
ax.set(xlim=[1,m], ylim=[0,1], xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$')
ax.legend(loc=4)
plt.savefig('SinVsRedBeta.pdf')
plt.show()  

n = 50
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, title=r'Projection errors for various solutions of $u_h(a(y))$ and average')

cp = sns.color_palette("hls", 8)
#plt.semilogy(range(1,n), dist_sin[0, 1:], ':', color=cp[0], linewidth=1, label=r'Sinusois basis: proj error single vec')
#for i, v in enumerate(soln_col[1:]):
#    plt.semilogy(range(1,n), dist_sin[i, 1:], ':', color=cp[0], linewidth=1)
plt.semilogy(range(1,n), dist_sin[:,1:].mean(axis=0), color=cp[0], label=r'Sinusoid basis: average projection error')
    
#plt.semilogy(range(1,n), dist_red[0, 1:], ':', color=cp[1], linewidth=1, label=r'Reduced basis: proj error single vec')
#for i, v in enumerate(soln_col[1:]):
#    plt.semilogy(range(1,n), dist_red[i, 1:], ':', color=cp[1], linewidth=1)
plt.semilogy(range(1,n), dist_red[:,1:].mean(axis=0), color=cp[1], label=r'Reduced basis: average projection error')
ax.set(xlim=[1,n-1], xlabel=r'$n$', ylabel=r'$||v - P_{V_n} v ||$')
ax.legend(loc=1)
plt.savefig('SinVsRedProjErr.pdf')
plt.show()  



