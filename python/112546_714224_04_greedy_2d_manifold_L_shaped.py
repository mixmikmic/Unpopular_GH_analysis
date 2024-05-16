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

def make_2d_param_soln(points, fem_div, a_bar=1.0, c=0.5, f=1.0, verbose=False):
    
    solns = []
    fields = []

    for p in points:
        field = pat.PWConstantSqDyadicL2(a_bar + c * np.array([[p[0], p[0]],[p[0], p[1]]]))
        fields.append(field)
        # Then the fem solver (there a faster way to do this all at once? This will be huge...
        fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = field, f = 1)
        fem_solver.solve()
        solns.append(fem_solver.u)
        
    return solns, fields

fem_div = 7

a_bar = 0.1
c = 2.0

np.random.seed(2)

y = np.array([[0.8, 0.1]])
print(y[0])
u, a = make_2d_param_soln(y, fem_div, a_bar=a_bar, c=c)
u = u[0]
a = a[0]

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
a.plot(ax, title='Example field $a(y)$ with $y\in\mathbb{R}^2$')
ax = fig.add_subplot(1, 2, 2, projection='3d')
u.plot(ax, title='FEM solution $u_h(a(y))$')
plt.show()

# local_width is the width of the measurement squares in terms of FEM mesh squares
width_div = 1
local_width = 2**width_div
spacing_div = 5

Wm_reg, Wloc_reg = pat.make_local_avg_grid_basis(width_div, spacing_div, fem_div, return_map=True)
Wm_reg = Wm_reg.orthonormalise()

m = Wm_reg.n
print('m =', m)

# We make the ambient spaces for Wm and Vn
np.random.seed(2)

Wm_rand, Wloc_rand = pat.make_pw_local_avg_random_basis(m=m, div=fem_div, width=local_width, return_map=True)
Wm_rand = Wm_rand.orthonormalise()

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 2, 1)
sns.heatmap(Wloc_rand.values, xticklabels=False, yticklabels=False, cbar=False, ax=ax)
ax.set_title('Random measurement locations')
ax = fig.add_subplot(1, 2, 2)
sns.heatmap(Wloc_reg.values, xticklabels=False, yticklabels=False, cbar=False, ax=ax)
ax.set_title('Regular measurement locations')
plt.plot()

dict_N = 50
dict_grid = np.linspace(0.0, 1.0, dict_N, endpoint=False)
y1s, y2s = np.meshgrid(dict_grid, dict_grid)

y1s = y1s.flatten()
y2s = y2s.flatten()

dict_ys = np.stack([y1s, y2s]).T

dictionary, dictionary_fields = make_2d_param_soln(dict_ys, fem_div, a_bar=a_bar, c=c)

greedy_algs = [pat.GreedyApprox(dictionary, Vn=pat.PWBasis(), verbose=True),
pat.MeasBasedOMP(dictionary, u, Wm_reg, Vn=pat.PWBasis(), verbose=True),
pat.MeasBasedPP(dictionary, u, Wm_reg, Vn=pat.PWBasis(), verbose=True),
pat.MeasBasedOMP(dictionary, u, Wm_rand, Vn=pat.PWBasis(), verbose=True),
pat.MeasBasedPP(dictionary, u, Wm_rand, Vn=pat.PWBasis(), verbose=True)]

greedy_algs_labels = ['Plain greedy', 
                      'Reg grid meas based OMP', 'Reg grid meas based PP', 
                      'Rand meas based OMP', 'Rand meas based PP',]

for g, l in zip(greedy_algs, greedy_algs_labels):
    print('Constructing ' + l)
    g.construct_to_n(m)

for i, greedy in enumerate(greedy_algs):
    ps = dict_ys[np.array(greedy.dict_sel, dtype=np.int32), :]
    print(greedy_algs_labels[i])
    print(ps)

sns.set_palette('hls', len(greedy_algs))
sns.set_style('whitegrid')

fig = plt.figure(figsize=(7,7))

for i, greedy in enumerate(greedy_algs):
    labels = ['{0} point {1}'.format(greedy_algs_labels[i], j) for j in range(greedy.n)] 
    
    ps = dict_ys[np.array(greedy.dict_sel, dtype=np.int32), :]
    
    plt.scatter(ps[:, 0], ps[:, 1], marker='o', label=greedy_algs_labels[i])

    for label, x, y in zip(labels, ps[:, 0], ps[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(-20, 20), textcoords='offset points', ha='right', va='bottom')

plt.legend()
plt.show()

fig = plt.figure(figsize=(15,12))
for i, v in enumerate(g.Vn.vecs):
    
    ax = fig.add_subplot(3, 3, i+1, projection='3d')
    v.plot(ax, title=r'$\phi_{{{0}}}$'.format(i+1))

plt.show()

lam, V = np.linalg.eigh(g.Vn.G)
print(lam[::-1])

fig = plt.figure(figsize=(15,12))
for i, v in enumerate(V.T[::-1]):
    
    vec = g.Vn.reconstruct(v)
    print(v)
    ax = fig.add_subplot(3, 3, i+1, projection='3d')
    vec.plot(ax, title='Eigenvector {0}'.format(i+1))

plt.show()



