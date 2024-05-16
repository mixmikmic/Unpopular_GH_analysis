import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")

import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels

#datadir = os.path.join("..", "..", "..", "..", "..", "Data")
datadir = os.path.join("/media", "disk", "Data")
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

def add_random_noise(points):
    ts = points.timestamps + np.random.random(size=points.timestamps.shape) * 60 * 1000 * np.timedelta64(1,"ms")
    ts = np.sort(ts)
    return points.from_coords(ts, points.xcoords, points.ycoords)

trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=1, timeunit=datetime.timedelta(days=1))
trainer.data = add_random_noise(points)
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)

model

pred = trainer.prediction_from_background(model)

fig, ax = plt.subplots(figsize=(10,6))

ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
mappable = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=ax)
ax.set_title("Estimated background rate")
None

np.max(model.mu), np.min(model.mu)

24 * 60 / model.omega

trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=1, timeunit=datetime.timedelta(days=1))
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=False)

model

24 * 60 / model.omega

sides = ["Far North", "Northwest", "North", "West", "Central",
    "South", "Southwest", "Far Southwest", "Far Southeast"]

def load(side):
    geo = open_cp.sources.chicago.get_side(side)
    grid = open_cp.data.Grid(150, 150, 0, 0)
    grid = open_cp.geometry.mask_grid_by_intersection(geo, grid)
    mask = (all_points.timestamps >= np.datetime64("2010-01-01")) & (all_points.timestamps < np.datetime64("2011-01-01"))
    points = all_points[mask]
    points = open_cp.geometry.intersect_timed_points(points, geo)
    return grid, points

def train(grid, points):
    trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=1)
    trainer.data = add_random_noise(points)
    model = trainer.train(datetime.datetime(2011,1,1), iterations=50)
    return model

for side in sides:
    model = train(*load(side))
    print(side, model.theta, 1/model.omega, np.max(model.mu))

grid, points = load("South")

trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=1)
trainer.data = add_random_noise(points)
model = trainer.train(datetime.datetime(2011,1,1), iterations=50)
model

model = trainer.train(datetime.datetime(2011,1,1), iterations=100)
model

model = trainer.train(datetime.datetime(2011,1,1), iterations=200)
model

cutoff = [0.1, 0.2, 0.5, 1, 1.5, 2]
lookup = {}
for c in cutoff:
    trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=c)
    trainer.data = add_random_noise(points)
    model = trainer.train(datetime.datetime(2011,1,1), iterations=100)
    lookup[c] = model

lookup

pred = trainer.prediction_from_background(lookup[0.1])
pred.mask_with(grid)
pred = pred.renormalise()

pred1 = trainer.prediction_from_background(lookup[0.5])
pred1.mask_with(grid)
pred1 = pred1.renormalise()

np.max(np.abs(pred.intensity_matrix - pred1.intensity_matrix))





