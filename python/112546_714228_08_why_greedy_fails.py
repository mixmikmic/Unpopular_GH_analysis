import numpy as np
import scipy as sp
import pandas as pd
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

import sys
sys.path.append("../../")
import pyApproxTools as pat
importlib.reload(pat)

get_ipython().magic('matplotlib inline')

results_file = '../../scripts/greedy_Vn/results/07_greedy_Vn_stats.npy'
stats = np.load(results_file)

g = pat.PWBasis(file_name = "../../scripts/greedy_Vn/results/PlainGreedy_Basis.npz")
r = pat.PWBasis(file_name = "../../scripts/greedy_Vn/results/Reduced_Basis.npz")
p = pat.PWBasis(file_name = "../../scripts/greedy_Vn/results/PCA_Basis.npz")

fig = plt.figure(figsize=(15, 15))
for i, v in enumerate(g.vecs[:16]):
    ax = fig.add_subplot(4, 4, i+1, projection='3d')
    v.plot(ax)
fig.savefig('GreedyPlots.pdf')
plt.show()


fig = plt.figure(figsize=(15, 15))
for i, v in enumerate(r.vecs[:16]):
    ax = fig.add_subplot(4, 4, i+1, projection='3d')
    v.plot(ax)
fig.savefig('ReducedBasisPlots.pdf')
plt.show()

generic_Vns_labels = ['Sinusoid basis', 'Reduced basis', 'Plain greedy', 'PCA']
adapted_Vns_labels = ['Meas based OMP', 'Meas based PP']
Wm_labels = ['Reg grid', 'Random']

sns.set_palette('hls', len(generic_Vns_labels) + len(adapted_Vns_labels))
cp = sns.color_palette()
sns.set_style('whitegrid')

lss = ['-', '--']
lws = [2,1]

axs = []
fig = plt.figure(figsize=(15, 8))
axs.append(fig.add_subplot(1, 2, 1, title='Projection error $\| u_h - P_{V_n} u_h \|_{H_0^1}$, Wm: Reg grid, with 100% CI'))
axs[-1].set(yscale="log", xlabel='$n$')
axs.append(fig.add_subplot(1, 2, 2, title=r'inf-sup condition $\beta(W_m, V_n)$'))
axs[-1].set(yscale="log", xlabel='$n$')

i = 0
Wm_label = Wm_labels[0]
for j, Vn_label in enumerate(generic_Vns_labels):

    Vn_n = np.where(np.isclose((~np.isclose(stats[0, i, j, :, :], 0.0)).sum(axis=0), stats.shape[3]))[-1][-1]

    label = 'Wm: ' + Wm_label + ' Vn: ' + Vn_label

    plt.sca(axs[0])
    sns.tsplot(stats[1, i, j, :, 2:Vn_n], range(2, Vn_n), ls=lss[i], lw=lws[i],color=cp[j], ci=[100])
    plt.plot(range(2, Vn_n), stats[1, i, j, :, 2:Vn_n].mean(axis=0), label=label, ls=lss[i], lw=lws[i], color=cp[j])
    plt.legend(loc=3)

    plt.sca(axs[1])
    plt.plot(range(2, Vn_n), stats[2, i, j, 0, 2:Vn_n], lss[i], label=label, lw=lws[i], color=cp[j])
    plt.legend(loc=3)

for j_i, Vn_label in enumerate(adapted_Vns_labels):
    j = j_i + len(generic_Vns_labels)

    label = 'Wm: ' + Wm_label + ' Vn: ' + Vn_label

    Vn_n = np.where(np.isclose((~np.isclose(stats[0, i, j, :, :], 0.0)).sum(axis=0), stats.shape[3]))[-1][-1]

    plt.sca(axs[0])
    sns.tsplot(stats[1, i, j, :, 2:Vn_n], range(2, Vn_n), ls=lss[i], lw=lws[i], color=cp[j], ci=[100])
    plt.plot(range(2, Vn_n), stats[1, i, j, :, 2:Vn_n].mean(axis=0), label=label, ls=lss[i], lw=lws[i], color=cp[j])        
    plt.legend(loc=3)

    plt.sca(axs[1])
    plt.plot(range(2, Vn_n), stats[2, i, j, 0, 2:Vn_n], lss[i], label=label, lw=lws[i], color=cp[j])
    plt.legend(loc=3)

plt.show()

p.dot(g.vecs[0])

def make_soln(points, fem_div, field_div, a_bar=1.0, c=0.5, f=1.0, verbose=False):
    
    solns = []
    fields = []

    for p in points:
        field = pat.PWConstantSqDyadicL2(a_bar + c * p.reshape((2**field_div,2**field_div)))
        fields.append(field)
        # Then the fem solver (there a faster way to do this all at once? This will be huge...
        fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = field, f = 1)
        fem_solver.solve()
        solns.append(fem_solver.u)
        
    return solns, fields

fem_div = 7

a_bar = 1.0
c = 0.9
field_div = 2
side_n = 2**field_div

def make_PCA(N = 1e3):

    np.random.seed(1)
    dict_basis_small, dict_fields = pat.make_pw_reduced_basis(N, field_div, fem_div, a_bar=a_bar, c=c, f=1.0, verbose=False)
    dict_basis_small.make_grammian()

    cent = dict_basis_small.reconstruct(np.ones(N) / N)

    import copy

    cent_vecs = copy.deepcopy(dict_basis_small.vecs)
    for i in range(len(cent_vecs)):
        cent_vecs[i] = cent_vecs[i] - cent

    dict_basis_small_cent = pat.PWBasis(cent_vecs)
    dict_basis_small_cent.make_grammian()
    
    lam, V = sp.linalg.eigh(dict_basis_small_cent.G)
    lams = np.sqrt(lam[np.abs(lam) > 1e-10][::-1])
    n = len(lams)

    PCA_vecs = []
    for i, v in enumerate(np.flip(V.T, axis=0)[:n]):
        vec = dict_basis_small_cent.reconstruct(v)
        PCA_vecs.append(vec / lams[i])

    return pat.PWBasis(PCA_vecs), lams

Vn_PCA_small_dict, small_lams = make_PCA(int(1e2))
Vn_PCA_mid_dict, mid_lams = make_PCA(int(5e2))
Vn_PCA_big_dict, big_lams = make_PCA(int(1e3))

# Now plot the rep of g.vecs[0] in terms of the PCA basis...

plt.semilogy(small_lams)
plt.semilogy(mid_lams)
plt.semilogy(big_lams)
plt.show()

Vn_PCA = Vn_PCA_big_dict
lams = big_lams
n = Vn_PCA.n

plt.semilogy(np.abs(Vn_PCA.dot(g.vecs[0]/g.vecs[0].norm())))
plt.semilogy(np.abs(Vn_PCA.dot(r.vecs[0]/r.vecs[0].norm())))
#plt.semilogy(np.abs(Vn_PCA.dot(g.vecs[1]/g.vecs[1].norm())))
#plt.semilogy(np.abs(Vn_PCA.dot(g.vecs[2]/g.vecs[2].norm())))
plt.semilogy(lams)
plt.show()

g_comp = Vn_PCA.dot(g.vecs[0]/g.vecs[0].norm())
r_comp = Vn_PCA.dot(r.vecs[0]/r.vecs[0].norm())

# Look at this
print(np.linalg.norm(g_comp[:3*n//4]))
print(np.linalg.norm(r_comp[:3*n//4]))
print(np.linalg.norm(g_comp[3*n//4:]))
print(np.linalg.norm(r_comp[3*n//4:]))

h_lev = 2 # The level the hat basis goes to

first_hat = pat.PWLinearSqDyadicH1(div = 1)
first_hat.values[1,1] = 1
first_hat = first_hat / first_hat.norm()

hat_basis = pat.PWBasis(vecs=[first_hat.interpolate(h_lev) / first_hat.norm()])

# Linear fill:
for l in range(1,h_lev):
    for i in range(2**(l+1)-1):
        for j in range(2**(l+1)-1):
            
            h = pat.PWLinearSqDyadicH1(div = l+1)
            h.values[i + 1, j + 1] = 1
            hat_basis.add_vector(h.interpolate(h_lev) / h.norm())
            
hat_basis.make_grammian()

print(np.linalg.cond(hat_basis.G))
print(hat_basis.G.shape)
print(np.linalg.matrix_rank(hat_basis.G))
print((hat_basis.vecs[0].values.shape[0]-2)**2)

for vec in hat_basis.vecs:
    print(vec.values)





