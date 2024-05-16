import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")

import opencrimedata.chicago
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime
import open_cp.predictors
import open_cp.kernels

#datadir = os.path.join("/media", "disk", "Data")
datadir = os.path.join("..", "..", "..", "..", "..", "Data")
with lzma.open(os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz"), "rt") as file:
    all_points = opencrimedata.chicago.load_to_open_cp(file, "BURGLARY")

open_cp.sources.chicago.set_data_directory(datadir)
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

#def add_random_noise(points):
#    ts = points.timestamps + np.random.random(size=points.timestamps.shape) * 60 * 1000 * np.timedelta64(1,"ms")
#    ts = np.sort(ts)
#    return points.from_coords(ts, points.xcoords, points.ycoords)

def add_random_noise(points):
    return points

trainer = sepp.sepp_grid.ExpDecayTrainer(grid)
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

trainer = sepp.sepp_grid.ExpDecayTrainer(grid)
trainer.data = add_random_noise(points)
model = trainer.train(datetime.datetime(2017,1,1), iterations=50, use_fast=False)

model

24 * 60 / model.omega

import open_cp.seppexp

trainer = open_cp.seppexp.SEPPTrainer(grid=grid)
trainer.data = add_random_noise(points)
predictor = trainer.train(iterations=50)

predictor.theta, predictor.omega * 60 * 24

predictor = trainer.train(iterations=50, use_corrected=True)

predictor.theta, predictor.omega * 60 * 24

pts = points.bin_timestamps(np.datetime64("2017-01-01T12:00"), np.timedelta64(1, "D"))

trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
trainer.data = pts
model = trainer.train(datetime.datetime(2017,1,1), iterations=200, use_fast=False)
model

1 / model.omega

trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
trainer.data = points.bin_timestamps(np.datetime64("2017-01-01T12:00"), np.timedelta64(12, "h"))
model = trainer.train(datetime.datetime(2017,1,2, 0,0), iterations=200, use_fast=False)
model

trainer.data.time_range

cells, _ = trainer.make_points(datetime.datetime(2017,1,2, 0,0))
min([np.min(x) for x in cells.flat if len(x)>0]), max([np.max(x) for x in cells.flat if len(x)>0])

trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
trainer.data = points.bin_timestamps(np.datetime64("2017-01-01T06:00"), np.timedelta64(12, "h"))
model = trainer.train(datetime.datetime(2017,1,1), iterations=200, use_fast=False)
model

trainer.data.time_range

cells, _ = trainer.make_points(datetime.datetime(2017,1,1))
min([np.min(x) for x in cells.flat if len(x)>0]), max([np.max(x) for x in cells.flat if len(x)>0])

ts = points.timestamps - np.timedelta64(6, "h")
pts = open_cp.data.TimedPoints(ts, points.coords)

trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
trainer.data = pts.bin_timestamps(np.datetime64("2017-01-01T06:00"), np.timedelta64(12, "h"))
model = trainer.train(datetime.datetime(2017,1,1), iterations=200, use_fast=False)
model

trainer.data.time_range

cells, _ = trainer.make_points(datetime.datetime(2017,1,1))
min([np.min(x) for x in cells.flat if len(x)>0]), max([np.max(x) for x in cells.flat if len(x)>0])

for hour in range(24):
    trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
    trainer.data = points.bin_timestamps(np.datetime64("2017-01-01T00:00")
                        + np.timedelta64(hour, "h"), np.timedelta64(1, "D"))
    model = trainer.train(datetime.datetime(2017,1,2, 0,0), iterations=50, use_fast=False)
    print(hour, model)

by_hour = {}
for hour in range(24):
    trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
    trainer.data = points.bin_timestamps(np.datetime64("2017-01-01T00:00")
                        + np.timedelta64(hour, "h"), np.timedelta64(12, "h"))
    model = trainer.train(datetime.datetime(2017,1,2, 0,0), iterations=100, use_fast=False)
    print(hour, model)
    by_hour[hour] = model

fig, ax = plt.subplots(figsize=(16, 5))
x = list(by_hour.keys())
x.sort()
y = [by_hour[t].theta for t in x]
ax.plot(x,y)

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
    trainer = sepp.sepp_grid.ExpDecayTrainer(grid)
    trainer.data = add_random_noise(points)
    model = trainer.train(datetime.datetime(2011,1,1), iterations=50)
    return model

for side in sides:
    model = train(*load(side))
    print(side, model.theta, 24*60/model.omega, np.max(model.mu))

def train(grid, points):
    trainer = sepp.sepp_grid.ExpDecayTrainer(grid, allow_repeats=True)
    trainer.data = points.bin_timestamps(np.datetime64("2017-01-01T12:00"), np.timedelta64(12, "h"))
    model = trainer.train(datetime.datetime(2011,1,1), iterations=50, use_fast=False)
    return model

for side in sides:
    model = train(*load(side))
    print(side, model.theta, 24*60/model.omega, np.max(model.mu))



