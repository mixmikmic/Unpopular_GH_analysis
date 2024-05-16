import numpy as np
import scipy as sp
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append("../../")
import pyApproxTools as pat
importlib.reload(pat)

get_ipython().magic('matplotlib inline')

N = 1e3
dictionary = pat.make_unif_dictionary(N)

ns = [10,20,40]
np.random.seed(3)
#n = 20
m = 200
bs_omp = np.zeros((len(ns), m))
bs_rand = np.zeros((len(ns), m))

Vn = pat.make_sin_basis(ns[-1])
Wms_omp = []
Wms_rand = []

for j, n in enumerate(ns):

    gbc = pat.CollectiveOMP(dictionary, Vn.subspace(slice(0,n)), verbose=True)
    Wm_omp = gbc.construct_to_m(m)
    Wms_omp.append(Wm_omp)
    Wm_omp_o = Wm_omp.orthonormalise()

    Wm_rand = pat.make_random_delta_basis(m)
    Wms_rand.append(Wm_rand)
    Wm_rand_o = Wm_rand.orthonormalise()

    BP_omp = pat.BasisPair(Wm_omp_o, Vn)
    BP_rand = pat.BasisPair(Wm_rand_o, Vn)
    for i in range(n, m):
        print('FB step ' + str(i))
        BP_omp_s = BP_omp.subspace(Wm_indices=slice(0,i), Vn_indices=slice(0,n)) #pat.BasisPair(Wm_omp_o.subspace(slice(0,i)), Vn.subspace(slice(0,n)))
        FB_omp = BP_omp_s.make_favorable_basis()
        bs_omp[j, i] = FB_omp.beta()

        BP_rand_s = BP_rand.subspace(Wm_indices=slice(0,i), Vn_indices=slice(0,n)) #pat.BasisPair(Wm_rand_o.subspace(slice(0,i)), Vn.subspace(slice(0,n)))
        FB_rand = BP_rand_s.make_favorable_basis()
        bs_rand[j, i] = FB_rand.beta()

sns.set_palette("deep")
cp = sns.color_palette()

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title='beta(Vn, Wm) against m for various n')#, title=r'$\beta(V_n, W_m)$ against $m$ for various $n$')

for i, n in enumerate(ns):
    plt.plot(range(m), bs_omp[i, :], label='omp Wm for n={0}'.format(n))#r'OMP constructed $W_m$, $n={{{0}}}$'.format(n))
    plt.plot(range(m), bs_rand[i, :], label='random Wm for n={0}'.format(n))#r'Random $W_m$, $n={{{0}}}$'.format(n))

ax.set(xlabel='m', ylabel='beta(Vn, Wm)')#r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=4)
plt.show()

sns.set_palette("deep")
cp = sns.color_palette()

Wm_omp = Wms_omp[1]
Vn = Vn.subspace(slice(0, 20))
b_omp = bs_omp[1,:]
b_rand = bs_rand[1,:]

n=20
m=200

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$'.format(n))

plt.plot(range(n,m), b_omp[n:], label=r'OMP constructed $W_m$')
plt.plot(range(n,m), b_rand[n:], label=r'Random $W_m$')

ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=2)
plt.show()


# Plot the evaluation points in the Wm_rand basis 
# (note that the basis is infact orthonormalised so this isn't *quite* an accurate picture)
Wm_points = [vec.elements.values_array()[0].keys_array() for vec in Wm_omp.vecs]

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$ for OMP basis, with eval points'.format(n))
ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$ and point locations')
plt.plot(range(n,n+40), b_omp[20:60], color=cp[0], label=r'$\beta(V_n, W_m)$ for OMP $W_m$')

plt.plot(n * np.ones(n-1), Wm_points[:n-1], 'o', color=cp[4], markersize=4, label='eval point')
plt.plot(n, Wm_points[n-1], 'o', color=cp[2], markersize=6, label='New eval point')
for m_plot in range(n, n+40-1):
    plt.plot((m_plot+1) * np.ones(m_plot), Wm_points[:m_plot], 'o', color=cp[4], markersize=4)
    plt.plot(m_plot+1, Wm_points[m_plot], 'o', color=cp[2], markersize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# Plot the evaluation points in the Wm_rand basis 
# (note that the basis is infact orthonormalised so this isn't *quite* an accurate picture)
Wm_points = [vec.elements.values_array()[0].keys_array() for vec in Wm_rand.vecs]
Wm_o_coeffs = [vec.elements.values_array()[0].values_array() for vec in Wm_rand_o.vecs]

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$ for random basis, with eval points'.format(n))
ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$ and point locations')
plt.plot(range(n,n+40), b_rand[20:60], color=cp[1], label=r'$\beta(V_n, W_m)$ for random $W_m$')

plt.plot(n * np.ones(n-1), Wm_points[:n-1], 'o', color=cp[4], markersize=4, label='eval point')
plt.plot(n, Wm_points[n-1], 'o', color=cp[2], markersize=6, label='New eval point')
for m_plot in range(n, n+40-1):
    plt.plot((m_plot+1) * np.ones(m_plot), Wm_points[:m_plot], 'o', color=cp[4], markersize=4)
    plt.plot(m_plot+1, Wm_points[m_plot], 'o', color=cp[2], markersize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

bs_unif_int = np.zeros((len(ns), m))
Vn = pat.make_sin_basis(ns[-1])

Wms_unif_int = []

for j, n in enumerate(ns):
    for i in range(n, m):
        
        Wm_unif_int = pat.Basis([pat.FuncVector(params=[[x]],coeffs=[[1.0]],funcs=['H1UIDelta']) for x in np.linspace(0.0, 1.0, i, endpoint=False)+0.5/i])
        Wm_unif_int_o = Wm_unif_int.orthonormalise()

        BP_ui = pat.BasisPair(Wm_unif_int_o, Vn.subspace(slice(0,n)))
        FB_ui = BP_ui.make_favorable_basis()
        bs_unif_int[j, i] = FB_ui.beta()

n = ns[1]
Wm_omp = Wms_omp[1]
Vn = Vn.subspace(slice(0, 20))
b_omp = bs_omp[1,:]
b_rand = bs_rand[1,:]
b_ui = bs_unif_int[1,:]

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$'.format(n))

plt.plot(range(n,m), b_omp[n:], label=r'OMP constructed $W_m$')
plt.plot(range(n,m), b_rand[n:], label=r'Random $W_m$')
plt.plot(range(n,m), b_ui[n:], label=r'Uniformly spaced $W_m$')

ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=2)
plt.show()

m=200
ns = [5, 10, 20, 40]#, 100]
bs_unif = np.zeros((len(ns), m))
bs_rand = np.zeros((len(ns), m))
bs_arb = np.zeros((len(ns), m))

gammas = np.arange(0., 1.1, 0.1)
m_gammas_unif = np.zeros((len(ns), len(gammas)))
m_gammas_rand = np.zeros((len(ns), len(gammas)))
m_gammas_arb = np.zeros((len(ns), len(gammas)))

for j, n in enumerate(ns):
    Vn = pat.make_sin_basis(n)
    
    omp_unif_x = np.load('omp_x_unif_{0}_10000.npy'.format(n))
    Wm_omp_unif = pat.Basis(vecs=[pat.FuncVector([[x]], [[1.0]], ['H1UIDelta']) for x in omp_unif_x])
    Wm_omp_unif_o = Wm_omp_unif.orthonormalise()

    omp_rand_x = np.load('omp_x_rand_{0}_10000.npy'.format(n))
    Wm_omp_rand = pat.Basis(vecs=[pat.Vector([[x]], [[1.0]], ['H1UIDelta']) for x in omp_rand_x])
    Wm_omp_rand_o = Wm_omp_rand.orthonormalise()

    Wm_arb = pat.make_random_delta_basis(m)
    Wm_arb_o = Wm_arb.orthonormalise()
    
    for i in range(n, m):
        BP_unif = pat.BasisPair(Wm_omp_unif_o.subspace(slice(0,i)), Vn)
        FB_unif = BP_unif.make_favorable_basis()
        bs_unif[j,i] = FB_unif.beta()

        BP_rand = pat.BasisPair(Wm_omp_rand_o.subspace(slice(0,i)), Vn)
        FB_rand = BP_rand.make_favorable_basis()
        bs_rand[j,i] = FB_rand.beta()
    
        BP_arb = pat.BasisPair(Wm_arb_o.subspace(slice(0,i)), Vn)
        FB_arb = BP_arb.make_favorable_basis()
        bs_arb[j,i] = FB_arb.beta()
    
    # Make the pivot data - the minimum m to reach some beta
    for i, gamma in enumerate(gammas):
        
        m_gammas_unif[j, i] = np.searchsorted(bs_unif[j,:], gamma)
        m_gammas_rand[j, i] = np.searchsorted(bs_rand[j,:], gamma)
        m_gammas_arb[j, i] = np.searchsorted(bs_arb[j,:], gamma)
        

sns.set_palette("deep")
cp = sns.color_palette()

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ for OMP with large dictionary ($N=10^6$)')

for i, n in enumerate(ns):
    
    plt.plot(range(m), bs_unif[i, :], label=r'OMP unif dict', color=cp[i])
    plt.plot(range(m), bs_rand[i, :], ':', label=r'OMP rand dict', color=cp[i])
    plt.plot(range(m), bs_arb[i, :], '--', label=r'Random $W_m$', color=cp[i])
    
ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=2)
plt.show()

"""THIS PLOT BELOW IS INTERESTING BUT CONFUSING: COMMENTED OUT FOR NOW
axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'Minimum $m$ to attain $\gamma$'.format(n))

for i, n in enumerate(ns):
    
    plt.plot(gammas, m_gammas_unif[i, :], label=r'OMP unif dict', color=cp[i])
    #plt.plot(gammas, m_gammas_rand[i, :], ':', label=r'OMP rand dict', color=cp[i])
    plt.plot(gammas, m_gammas_arb[i, :], '--', label=r'Random $W_m$', color=cp[i])
    
ax.set(xlabel=r'$\gamma$', ylabel=r'$\mathrm{argmin}\{m : \beta(V_n, W_m) > \gamma \}$')
plt.legend(loc=2)
plt.show()
"""

sns.set_palette("deep")
cp = sns.color_palette()

Wm_omp = Wms_omp[1]
Vn = Vn.subspace(slice(0, 20))
b_omp = bs_omp[1,:]
n=20

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$, comparing results for different dictionary sizes'.format(n))

plt.plot(range(m), b_omp[:], label=r'Small dictionary')
plt.plot(range(m), bs_unif[2, :], label=r'Large dictionary')

ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$')
plt.legend(loc=2)
plt.show()

# Plot the evaluation points in the Wm_omp basis - generated from a small dictionary
Wm_points = [vec.elements[0][0] for vec in Wm_omp.vecs]

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$ for small dictionary OMP basis, with eval points'.format(n))
ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$ and point locations')
plt.plot(range(n,n+40), b_omp[20:60], color=cp[1], label=r'$\beta(V_n, W_m)$ for OMP $W_m$')

plt.plot(n * np.ones(n-1), Wm_points[:n-1], 'o', color=cp[4], markersize=4, label='eval point')
plt.plot(n, Wm_points[n-1], 'o', color=cp[2], markersize=6, label='New eval point')
for m_plot in range(n, n+40-1):
    plt.plot((m_plot+1) * np.ones(m_plot), Wm_points[:m_plot], 'o', color=cp[4], markersize=4)
    plt.plot(m_plot+1, Wm_points[m_plot], 'o', color=cp[2], markersize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# Now for the Wm_omp_unif basis - generated from a large dictionary
omp_unif_x = np.load('omp_x_unif_{0}_10000.npy'.format(n))
Wm_omp_unif = pat.Basis(vecs=[omp.Vector([x], [1.0], ['H1delta']) for x in omp_unif_x])

Wm_points = [vec.elements[0][0] for vec in Wm_omp_unif.vecs]

axs = []
fig = plt.figure(figsize=(13, 9))
ax = fig.add_subplot(1, 1, 1, title=r'$\beta(V_n, W_m)$ against $m$ for $n={{{0}}}$ for large dictionary OMP basis, with eval points'.format(n))
ax.set(xlabel=r'$m$', ylabel=r'$\beta(V_n, W_m)$ and point locations')
plt.plot(range(n,n+40), bs_unif[2, 20:60], color=cp[1], label=r'$\beta(V_n, W_m)$ for OMP $W_m$')

plt.plot(n * np.ones(n-1), Wm_points[:n-1], 'o', color=cp[4], markersize=4, label='eval point')
plt.plot(n, Wm_points[n-1], 'o', color=cp[2], markersize=6, label='New eval point')
for m_plot in range(n, n+40-1):
    plt.plot((m_plot+1) * np.ones(m_plot), Wm_points[:m_plot], 'o', color=cp[4], markersize=4)
    plt.plot(m_plot+1, Wm_points[m_plot], 'o', color=cp[2], markersize=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

print(bs_unif[2, 19:30])



