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

def make_soln(points, fem_div, a_bar=1.0, c=0.5, f=1.0, verbose=False):
    
    solns = []
    fields = []

    for p in points:
        field = pat.PWConstantSqDyadicL2(a_bar + c * p.reshape((2,2)))
        fields.append(field)
        # Then the fem solver (there a faster way to do this all at once? This will be huge...
        fem_solver = pat.DyadicFEMSolver(div=fem_div, rand_field = field, f = 1)
        fem_solver.solve()
        solns.append(fem_solver.u)
        
    return solns, fields

fem_div = 8

a_bar = 0.1
c = 2.0

np.random.seed(2)

y = np.random.random((1,4))

u, a = make_soln(y, fem_div, a_bar=a_bar, c=c)
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
spacing_div = 4

Wm_reg, Wloc_reg = pat.make_local_avg_grid_basis(width_div, spacing_div, fem_div, return_map=True)
Wm_reg = Wm_reg.orthonormalise()

m = Wm_reg.n
print('m =', m)

# We make the ambient spaces for Wm and Vn
np.random.seed(2)

Wm_rand, Wloc_rand = pat.make_pw_local_avg_random_basis(m=m, div=fem_div, width=local_width, return_map=True)
Wm_rand = Wm_rand.orthonormalise()

fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(Wloc_reg.values, xticklabels=False, yticklabels=False, cbar=False, ax=ax)
fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(Wloc_rand.values, xticklabels=False, yticklabels=False, cbar=False, ax=ax)
ax.set_title('Measurement locations')
##plt.savefig('ddgrb_measurements.pdf')
plt.plot()

dict_N = 10
dict_grid = np.linspace(0.0, 1.0, dict_N, endpoint=False)
y1s, y2s, y3s, y4s = np.meshgrid(dict_grid, dict_grid, dict_grid, dict_grid)

y1s = y1s.flatten()
y2s = y2s.flatten()
y3s = y3s.flatten()
y4s = y4s.flatten()

dict_ys = np.stack([y1s, y2s, y3s, y4s]).T
print('Making dictionary of length', len(dict_ys))
dictionary, dictionary_fields = make_soln(dict_ys, fem_div, a_bar=a_bar, c=c)

g = pat.GreedyApprox(dictionary, Vn=pat.PWBasis(), verbose=True, remove=False)
g.construct_to_n(m)

mbg_rand = pat.MeasBasedGreedy(dictionary, Wm_rand.dot(u), Wm_rand, Vn=pat.PWBasis(), verbose=True, remove=False)
mbg_rand.construct_to_n(m)

mbg_reg = pat.MeasBasedGreedy(dictionary, Wm_reg.dot(u), Wm_reg, Vn=pat.PWBasis(), verbose=True, remove=False)
mbg_reg.construct_to_n(m)

mbgp_rand = pat.MeasBasedGreedyPerp(dictionary, Wm_rand.dot(u), Wm_rand, Vn=pat.PWBasis(), verbose=True, remove=False)
mbgp_rand.construct_to_n(m)

mbgp_reg = pat.MeasBasedGreedyPerp(dictionary, Wm_reg.dot(u), Wm_reg, Vn=pat.PWBasis(), verbose=True, remove=False)
mbgp_reg.construct_to_n(m)

#Vn_sin = pat.make_pw_sin_basis(div=fem_div)

greedys = [g, mbg_rand, mbg_reg,mbgp_rand, mbgp_reg]
g_labels = ['Plain', 'Meas., Wm random', 'Meas. Wm regular', 'Perp. Wm random', 'Perp. Wm regular']

for i, greedy in enumerate(greedys):

    ps = dict_ys[np.array(greedy.dict_sel, dtype=np.int32), :]
    print(g_labels[i])
    print(ps)

sns.set_palette('hls', len(greedys))
sns.set_style('whitegrid')

fig = plt.figure(figsize=(7,7))

for i, greedy in enumerate(greedys):
    labels = ['{0} point {1}'.format(g_labels[i], j) for j in range(greedy.n)] 
    
    ps = dict_ys[np.array(greedy.dict_sel, dtype=np.int32), :]
    
    plt.scatter(ps[:, 0], ps[:, 1], marker='o')

    for label, x, y in zip(labels, ps[:, 0], ps[:, 1]):
        plt.annotate(
            label, xy=(x, y), xytext=(-20, 20), textcoords='offset points', ha='right', va='bottom')

plt.show()

np.random.seed(2)

y = np.random.random((1,4))
y = np.array([[0.1, 0.3, 0.6, 0.9]])

u, a = make_soln(y, fem_div, a_bar=a_bar, c=c)

u0, a0 = make_soln(y.mean()*np.ones((1,4)), fem_div, a_bar=a_bar, c=c)
u1, a1 = make_soln(np.array([[1,1,1e16,1e16]]), fem_div, a_bar=0, c=1)
u2, a2 = make_soln(np.array([[1e16,1,1e16,1]]), fem_div, a_bar=0, c=1)
u3, a3 = make_soln(np.array([[1e16,1e16,1,1]]), fem_div, a_bar=0, c=1)
u4, a4 = make_soln(np.array([[1,1e16,1,1e16]]), fem_div, a_bar=0, c=1)

u5, a5 = make_soln(np.array([[1,1e16,1e16,1e16]]), fem_div, a_bar=0, c=1)
u6, a6 = make_soln(np.array([[1e16,1,1e16,1e16]]), fem_div, a_bar=0, c=1)
u7, a7 = make_soln(np.array([[1e16,1e16,1,1e16]]), fem_div, a_bar=0, c=1)
u8, a8 = make_soln(np.array([[1e16,1e16,1e16,1]]), fem_div, a_bar=0, c=1)

# The forgotten corner solutions...?
u9, a9   = make_soln(np.array([[1,1,1,1e16]]), fem_div, a_bar=0, c=1)
u10, a10 = make_soln(np.array([[1e16,1,1,1]]), fem_div, a_bar=0, c=1)
u11, a11 = make_soln(np.array([[1,1e16,1,1]]), fem_div, a_bar=0, c=1)
u12, a12 = make_soln(np.array([[1,1,1e16,1]]), fem_div, a_bar=0, c=1)

u = u[0]
us=[]
us.append(u0[0])
us.append(u1[0])
us.append(u2[0])
us.append(u3[0])
us.append(u4[0])
us.append(u5[0])
us.append(u6[0])
us.append(u7[0])
us.append(u8[0])
#us.append(u9[0])
#us.append(u10[0])
#us.append(u11[0])
#us.append(u12[0])

print(y)
print(a[0].values)

fig = plt.figure(figsize=(15, 20))
ax = fig.add_subplot(4, 4, 1, projection='3d')
u.plot(ax, title='$u$')
ax = fig.add_subplot(4, 4, 2, projection='3d')
a[0].plot(ax, title='$a$')
for i,v in enumerate(us):
    ax = fig.add_subplot(4, 4, i+3, projection='3d')
    v.plot(ax, title=r'$u_{{{0}}}$'.format(i))

plt.show()

M = np.vstack([v.values.flatten() for v in us])
w = u.values.flatten()

C = M @ M.T
g = M @ w

#cf = np.linalg.lstsq(M.T, w)[0]
#print(cf)
cf = np.linalg.solve(C, g)
print("Coefficients:", cf)

#print(M @ M.T)
lambdas, V =  np.linalg.eig(M @ M.T)
print("eigenvalues: ", lambdas)

u_rec = pat.PWLinearSqDyadicH1(us[0].values * cf[0])
for i,v in enumerate(us[1:]):
    u_rec += v * cf[i+1]

print((u - u_rec).values.max())
print(y)

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1, projection='3d')
(u-u_rec).plot(ax, title='$u$')
#ax = fig.add_subplot(2, 2, 2, projection='3d')
#corner.plot(ax, title='')
#ax = fig.add_subplot(2, 2, 3, projection='3d')
#u.plot(ax, title='')
#ax = fig.add_subplot(2, 2, 4, projection='3d')
#u5.plot(ax, title='')
plt.show()

us_basis = pat.PWBasis(us)

lambdas, V =  np.linalg.eig(M @ M.T)
print("eigenvalues: ", lambdas)


fig = plt.figure(figsize=(15, 20))

for i, v in enumerate(V.T):
    ax = fig.add_subplot(4, 4, i+1, projection='3d')
    us_basis.reconstruct(v).plot(ax, title='$u$')

