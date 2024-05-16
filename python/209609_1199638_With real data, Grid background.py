import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_fixed
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.naive

datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")

northside = open_cp.sources.chicago.get_side("North")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)

mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
points = open_cp.geometry.intersect_timed_points(points, northside)

fig, axes = plt.subplots(ncols=2, figsize=(16,10))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.scatter(points.xcoords, points.ycoords, marker="x", color="black", linewidth=1)

kernel = open_cp.kernels.GaussianBase(points.coords)
kernel.bandwidth = 300
kernel.covariance_matrix = [[1,0], [0,1]]
pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
None

tk = sepp.sepp_fixed.ExpTimeKernel(0.2)
sk = sepp.sepp_fixed.GaussianSpaceKernel(50)
trainer = sepp.sepp_fixed.GridTrainer(grid, tk, sk)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model

fig, axes = plt.subplots(ncols=2, figsize=(16,5))

for ax in axes:
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)

ax = axes[0]
pred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
m = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
cb = fig.colorbar(m, ax=ax)
ax.set_title("SEPP prediction background")

naive = open_cp.naive.CountingGridKernel(grid.xsize, grid.ysize, grid.region())
naive.data = points
pred = naive.predict().renormalise()
ax = axes[1]
m = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
cb = fig.colorbar(m, ax=ax)
ax.set_title("Naive background rate")

None

tk = sepp.sepp_fixed.ExpTimeKernel(0.2)
sk = sepp.sepp_fixed.GaussianSpaceKernel(50)
trainer = sepp.sepp_fixed.GridTrainer(grid, tk, sk)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.initial_model(T, data)
for _ in range(100):
    opt = trainer._optimiser(model, data)
    old_model = model
    model = opt.iterate()
    print(model, np.mean((model.mu - old_model.mu)**2), (model.theta - old_model.theta)**2)

omegas = np.linspace(0.02, 1, 20)
sigmas = [50] # [10, 25, 50, 100, 250]
models = dict()
for omega in omegas:
    for s in sigmas:
        tk = sepp.sepp_fixed.ExpTimeKernel(omega)
        sk = sepp.sepp_fixed.GaussianSpaceKernel(s)
        trainer = sepp.sepp_fixed.GridTrainer(grid, tk, sk)
        trainer.data = points
        models[(omega,s)] = trainer.train(datetime.datetime(2017,1,1), iterations=20)

fig, ax = plt.subplots(figsize=(8,6))

ax.plot([1/o for o in omegas], [models[(o,50)].theta for o in omegas])
ax.set(xlabel="$\omega^{-1}$")
#ax.plot(omegas, [models[(o,50)].theta for o in omegas])
#ax.set(xlabel="$\omega$")
ax.set(ylabel="$\theta$")

omegas_inv = np.linspace(1, 250, 150)
sigmas = [10, 25, 50, 100, 250]
models = dict()
for omega_inv in omegas_inv:
    for s in sigmas:
        tk = sepp.sepp_fixed.ExpTimeKernel(1 / omega_inv)
        sk = sepp.sepp_fixed.GaussianSpaceKernel(s)
        trainer = sepp.sepp_fixed.GridTrainer(grid, tk, sk)
        trainer.data = points
        models[(omega_inv,s)] = trainer.train(datetime.datetime(2017,1,1), iterations=20)

fig, ax = plt.subplots(figsize=(12,6))

for s in sigmas:
    ax.plot(omegas_inv, [models[(o,s)].theta for o in omegas_inv])
ax.legend(["$\sigma={}$".format(s) for s in sigmas])
ax.set(xlabel="$\omega^{-1}$")
ax.set(ylabel="$\\theta$")

all_models = []
for sigma, maxoi in zip([10, 25, 50, 100, 250], [500,200,50,20,6]):
    omegas_inv = np.linspace(1, maxoi, 50)
    models = []
    for omega_inv in omegas_inv:
        tk = sepp.sepp_fixed.ExpTimeKernel(1 / omega_inv)
        sk = sepp.sepp_fixed.GaussianSpaceKernel(sigma)
        trainer = sepp.sepp_fixed.GridTrainer(grid, tk, sk)
        trainer.data = points
        models.append( trainer.train(datetime.datetime(2017,1,1), iterations=20) )
    all_models.append((sigma, omegas_inv, models))

fig, axes = plt.subplots(ncols=5, figsize=(18,5))

for ax, (s, ois, models) in zip(axes, all_models):
    ax.plot(ois, [m.theta for m in models], color="black")
    ax.set(xlabel="$\omega^{-1}$")
    #ax.set(ylabel="$\\theta$")
    ax.set(title="$\sigma={}$".format(s))
fig.tight_layout()

fig.savefig("../fixed_grid_1.pdf")



