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
_pred = open_cp.predictors.grid_prediction_from_kernel_and_masked_grid(kernel, grid, samples=5)
ax = axes[1]
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
ax.pcolor(*_pred.mesh_data(), _pred.intensity_matrix, cmap="Greys", rasterized=True)
None

trainer = sepp.sepp_grid_space.Trainer2(grid, 20)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)
model

predictor = trainer.to_predictor(model)
predictor.data = trainer.data
pred1 = predictor.predict(datetime.datetime(2017,1,1))
pred2 = predictor.predict(datetime.datetime(2016,9,1))
pred3 = predictor.predict(datetime.datetime(2016,10,15))

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16,10))

for ax in axes.flat:
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)

bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)

for ax, pred in zip(axes.flat, [bpred, pred1, pred2, pred3]):
    m = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)
fig.tight_layout()

r0_range = np.exp(np.linspace(0, np.log(200), 100))
models = {}
for r0 in r0_range:
    trainer = sepp.sepp_grid_space.Trainer2(grid, r0)
    trainer.data = points
    models[r0] = trainer.train(datetime.datetime(2017,1,1), iterations=50)

fig, axes = plt.subplots(ncols=2, figsize=(16,3))

axes[0].plot(r0_range, [models[r].theta for r in r0_range], color="black")
axes[0].set(title="theta")
axes[1].plot(r0_range, [1/models[r].omega for r in r0_range], color="black")
axes[1].set(title="1 / omega")

fig.tight_layout()
fig.savefig("../varying_r0_no_g.pdf")

trainer = sepp.sepp_grid_space.Trainer2(grid, 300)
trainer.data = points

T, data = trainer.make_data()
model = trainer.initial_model(T, data)
model

for _ in range(50):
    model = sepp.sepp_grid_space.Model2(model.mu, model.T, model.grid, model.theta, 1/20, model.r0)
    opt = trainer._optimiser(model, data)
    model = opt.iterate()

model

bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
fig, ax = plt.subplots(figsize=(8,5))
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
cb = fig.colorbar(m, ax=ax)

backs = model.background(data)

trigs = [0]
for j in range(1, data.shape[-1]):
    deltas = data[:,j][:,None] - data[:,:j]
    trigs.append( np.sum(model.trigger(None, deltas)) )
trigs = np.asarray(trigs) / model.theta

np.sum(trigs / backs) - len(backs)

model.omega * np.exp(-model.omega) / (np.pi * model.r0**2)

T, data = trainer.make_data()
model = trainer.initial_model(T, data)
for _ in range(150):
    model = sepp.sepp_grid_space.Model2(model.mu, model.T, model.grid, 0.4, 1/20, model.r0)
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model

backs = model.background(data)
trigs = [0]
for j in range(1, data.shape[-1]):
    deltas = data[:,j][:,None] - data[:,:j]
    trigs.append( np.sum(model.trigger(None, deltas)) )
trigs = np.asarray(trigs) / model.theta

np.sum(trigs / backs) - len(backs)

bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
fig, ax = plt.subplots(figsize=(8,5))
ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
ax.set_aspect(1)
m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
cb = fig.colorbar(m, ax=ax)

w = trigs / backs
w = w[w>0]
def f(theta):
    return np.sum(1/(1/w + theta)) - len(trigs)
f(0), f(0.01), f(0.1), f(0.2)

theta = np.linspace(0.05, 0.2, 200)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(theta, [f(t) for t in theta])
ax.plot(theta, [0]*len(theta))



