import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, os
import open_cp.predictors
import open_cp.kernels
import open_cp.seppexp
import open_cp.naive
import open_cp.evaluation
import open_cp.logger
open_cp.logger.log_to_true_stdout()

datadir = os.path.join("..", "..", "..", "..", "..", "Data")
#datadir = os.path.join("/media", "disk", "Data")
open_cp.sources.chicago.set_data_directory(datadir)

with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
    all_points = open_cp.sources.chicago.load(file, "BURGLARY", type="all")

def load(side, start, end):
    geo = open_cp.sources.chicago.get_side(side)
    grid = open_cp.data.Grid(150, 150, 0, 0)
    grid = open_cp.geometry.mask_grid_by_intersection(geo, grid)
    mask = (all_points.timestamps >= start) & (all_points.timestamps < end)
    points = all_points[mask]
    points = open_cp.geometry.intersect_timed_points(points, geo)
    return grid, points, geo

grid, points, geo = load("South", np.datetime64("2010-01-01"), np.datetime64("2010-09-01"))

trainer = open_cp.seppexp.SEPPTrainer(grid=grid)
trainer.data = points
predictor = trainer.train(cutoff_time=np.datetime64("2011-09-01"), iterations=50)

back = predictor.background_prediction()
predictor.data = points
pred = predictor.predict(np.datetime64("2011-09-01T12:00"))
predictor.theta, predictor.omega * 60 * 24

npredictor = open_cp.naive.CountingGridKernel(grid.xsize, region=grid.region())
npredictor.data = points
naive = npredictor.predict()
naive.mask_with(grid)
naive = naive.renormalise()

fig, axes = plt.subplots(ncols=3, figsize=(16,6))

for ax in axes:
    ax.add_patch(descartes.PolygonPatch(geo, fc="none"))
    ax.set_aspect(1)

mappable = axes[0].pcolor(*back.mesh_data(), back.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=axes[0])
axes[0].set_title("Estimated background rate")

mappable = axes[1].pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=axes[1])
axes[1].set_title("Full prediction")

mat = pred.intensity_matrix - back.intensity_matrix
mappable = axes[2].pcolor(*pred.mesh_data(), mat, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=axes[2])
axes[2].set_title("Difference")

fig.tight_layout()

fig, axes = plt.subplots(ncols=3, figsize=(16,6))

for ax in axes:
    ax.add_patch(descartes.PolygonPatch(geo, fc="none"))
    ax.set_aspect(1)

nback = back.renormalise()
mappable = axes[0].pcolor(*nback.mesh_data(), nback.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=axes[0])
axes[0].set_title("Normalised background rate")

mappable = axes[1].pcolor(*naive.mesh_data(), naive.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=axes[1])
axes[1].set_title("'Naive' prediction")

mat = naive.intensity_matrix - nback.intensity_matrix
mappable = axes[2].pcolor(*pred.mesh_data(), mat, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=axes[2])
axes[2].set_title("Difference")

fig.tight_layout()

class SeppExpProvider(open_cp.evaluation.StandardPredictionProvider):
    def give_prediction(self, grid, points, time):
        trainer = open_cp.seppexp.SEPPTrainer(grid=grid)
        trainer.data = points
        predictor = trainer.train(cutoff_time=time, iterations=50)
        return predictor.background_prediction()

grid, points, geo = load("South", np.datetime64("2010-01-01"), np.datetime64("2011-01-01"))

provider = open_cp.evaluation.NaiveProvider(points, grid)
evaluator = open_cp.evaluation.HitRateEvaluator(provider)
evaluator.data = points
time_range = evaluator.time_range(datetime.datetime(2010,9,1),
        datetime.datetime(2010,12,31), datetime.timedelta(days=1))
result = evaluator.run(time_range, range(1,100))

provider = SeppExpProvider(points, grid)
evaluator = open_cp.evaluation.HitRateEvaluator(provider)
evaluator.data = points
time_range = evaluator.time_range(datetime.datetime(2010,9,1),
        datetime.datetime(2010,12,31), datetime.timedelta(days=1))
result1 = evaluator.run(time_range, range(1,100))

import pandas as pd

frame = pd.DataFrame(result.rates).T.describe().T
frame.head()

frame1 = pd.DataFrame(result1.rates).T.describe().T
frame1.head()

fig, ax = plt.subplots(figsize=(14,8))

ax.plot(frame["mean"]*100, label="naive")
ax.plot(frame1["mean"]*100, label="sepp")

import sepp.sepp_grid

class SeppExpProvider(open_cp.evaluation.StandardPredictionProvider):
    def give_prediction(self, grid, points, time):
        trainer = sepp.sepp_grid.ExpDecayTrainerWithCutoff(grid, cutoff=0.5)
        trainer.data = points # Add noise??
        model = trainer.train(time, iterations=50)
        return trainer.prediction_from_background(model)

grid, points, geo = load("South", np.datetime64("2010-01-01"), np.datetime64("2011-01-01"))

provider = open_cp.evaluation.NaiveProvider(points, grid)
evaluator = open_cp.evaluation.HitRateEvaluator(provider)
evaluator.data = points
time_range = evaluator.time_range(datetime.datetime(2010,9,1),
        datetime.datetime(2010,12,31), datetime.timedelta(days=1))
result = evaluator.run(time_range, range(1,100))

provider = SeppExpProvider(points, grid)
evaluator = open_cp.evaluation.HitRateEvaluator(provider)
evaluator.data = points
time_range = evaluator.time_range(datetime.datetime(2010,9,1),
        datetime.datetime(2010,12,31), datetime.timedelta(days=1))
result1 = evaluator.run(time_range, range(1,100))

import pandas as pd
frame = pd.DataFrame(result.rates).T.describe().T
frame1 = pd.DataFrame(result1.rates).T.describe().T

fig, ax = plt.subplots(figsize=(14,8))

ax.plot(frame["mean"]*100, label="naive")
ax.plot(frame1["mean"]*100, label="sepp")



