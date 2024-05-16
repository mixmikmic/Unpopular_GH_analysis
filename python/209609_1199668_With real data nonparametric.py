import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import sepp.grid_nonparam

import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")

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

trainer = sepp.grid_nonparam.NonParamTrainer(grid, bandwidth=1.5)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=False)

model

pred = trainer.prediction_from_background(model)

fig, axes = plt.subplots(ncols=2, figsize=(16,6))

ax = axes[0]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
mappable = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=ax)
ax.set_title("Estimated background rate")

ax = axes[1]
x = np.arange(10) * (trainer.time_unit / np.timedelta64(1,"D")) * model.bandwidth
#ax.scatter(x, model.alpha[:len(x)] * model.theta)
ax.set(xlabel="Days", ylabel="Rate", title="Trigger kernel")
ax.bar(x + (x[1] - x[0]) / 2, model.alpha[:len(x)] * model.theta / model.bandwidth,
    model.bandwidth, color="None", edgecolor="black")
#ax.bar(x + (x[1] - x[0]) / 2, model.trigger(None, x), model.bandwidth, color="None", edgecolor="black")
None

np.max(model.mu), np.min(model.mu)

bandwidths = [0.05, 0.15, 0.3, 1]
models = {}
for b in bandwidths:
    trainer = sepp.grid_nonparam.NonParamTrainer(grid, bandwidth=b)
    trainer.data = points
    models[b] = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=False)
    print(b, models[b])

fig, axes = plt.subplots(ncols=4, figsize=(16,4))

for ax, (b, model), s in zip(axes, models.items(), [600,200,100,30]):
    x = np.arange(s) * (trainer.time_unit / np.timedelta64(1,"D")) * model.bandwidth
    ax.set(xlabel="Days", ylabel="Rate", title="Trigger kernel, h={} days".format(b))
    ax.bar(x + (x[1] - x[0]) / 2, model.alpha[:len(x)] * model.theta / model.bandwidth,
           model.bandwidth, color="None", edgecolor="black")
fig.tight_layout()
fig.savefig("../north_trigger.pdf")

bandwidths = [0.05, 0.15, 0.3, 1]
models = {}
for b in bandwidths:
    trainer = sepp.grid_nonparam.NonParamTrainer(grid, bandwidth=b)
    trainer.data = points
    models[b] = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=True)
    print(b, models[b])

fig, axes = plt.subplots(ncols=4, figsize=(16,4))

for ax, (b, model), s in zip(axes, models.items(), [600,200,100,30]):
    x = np.arange(s) * (trainer.time_unit / np.timedelta64(1,"D")) * model.bandwidth
    ax.set(xlabel="Days", ylabel="Rate", title="Trigger kernel, h={} days".format(b))
    ax.bar(x + (x[1] - x[0]) / 2, model.alpha[:len(x)] * model.theta / model.bandwidth,
           model.bandwidth, color="None", edgecolor="black")
fig.tight_layout()

sides = ["Far North", "Northwest", "North", "West", "Central",
    "South", "Southwest", "Far Southwest", "Far Southeast"]

def load(side):
    geo = open_cp.sources.chicago.get_side(side)
    grid = open_cp.data.Grid(150, 150, 0, 0)
    grid = open_cp.geometry.mask_grid_by_intersection(geo, grid)
    mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
    points = all_points[mask]
    points = open_cp.geometry.intersect_timed_points(points, geo)
    return grid, points

bandwidths = [0.05, 0.15, 0.3, 1]

models = {}
for side in sides:
    grid, points = load(side)
    models[side] = {}
    for b in bandwidths:
        trainer = sepp.grid_nonparam.NonParamTrainer(grid, bandwidth=b)
        trainer.data = points
        try:
            models[side][b] = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=False)
        except ValueError as ex:
            #print("Failed because {} for {}/{}".format(ex, side, b))
            print("Failed: {}/{}".format(side, b))
            models[side][b] = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=True)

fig, axes = plt.subplots(ncols=4, nrows=len(sides), figsize=(16,20))

for side, axe in zip(sides, axes):
    for ax, bw, s in zip(axe, models[side], [900,300,150,50]):
        model = models[side][bw]
        x = np.arange(s) * (trainer.time_unit / np.timedelta64(1,"D")) * model.bandwidth
        ax.set(xlabel="Days", ylabel="Rate", title="{}, h={} days".format(side, bw))
        ax.bar(x + (x[1] - x[0]) / 2, model.alpha[:len(x)] * model.theta / model.bandwidth,
               model.bandwidth, color="None", edgecolor="black")

fig.tight_layout()







