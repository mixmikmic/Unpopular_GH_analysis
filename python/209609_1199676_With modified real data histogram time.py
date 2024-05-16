import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import sepp.sepp_grid_space
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")

import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels
import opencrimedata.chicago

datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
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

def plot(model, histlen):
    fig, axes = plt.subplots(ncols=2, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    ax = axes[1]
    x = np.arange(histlen) * model.bandwidth
    ax.bar(x + model.bandwidth/2, model.alpha_array[:len(x)] * model.theta / model.bandwidth,
           model.bandwidth, color="none", edgecolor="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    None

trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=1, use_fast=True)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model

plot(model, 30)

trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=0.5, use_fast=True)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model

plot(model, 40)

trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=0.1, use_fast=True)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model

plot(model, 40)

trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=0.8, use_fast=True)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model

plot(model, 40)

trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=1.2, use_fast=True)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model

plot(model, 35)

trainer = sepp.sepp_grid_space.Trainer3(grid, r0=20, bandwidth=1.3, use_fast=True)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model

plot(model, 35)

trainer = sepp.sepp_grid_space.Trainer3(grid, r0=100, bandwidth=1)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model

plot(model, 10)

trainer = sepp.sepp_grid_space.Trainer3(grid, r0=100, bandwidth=0.1)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model

plot(model, 30)

trainer = sepp.sepp_grid_space.Trainer3(grid, r0=250, bandwidth=1)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model

plot(model, 6)

trainer = sepp.sepp_grid_space.Trainer3(grid, r0=250, bandwidth=0.1)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model

plot(model, 60)





