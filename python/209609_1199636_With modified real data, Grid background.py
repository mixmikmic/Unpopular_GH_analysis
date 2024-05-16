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
import opencrimedata.chicago

datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")

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

all_models = []
for sigma, maxoi in zip([10, 25, 50, 100, 250], [200,30,20,20,6]):
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

fig.savefig("../fixed_grid_mod_1.pdf")



