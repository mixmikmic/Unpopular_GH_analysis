import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_full
import open_cp.logger
open_cp.logger.log_to_true_stdout("sepp")
open_cp.logger.log_to_true_stdout()
import open_cp.sources.chicago
import open_cp.geometry
import descartes
import pickle, lzma, datetime, collections, os
import open_cp.predictors
import sepp.kernels
import scipy.stats
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

from plotting_split import *

tk_time_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
tk_space_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(20)
opt_fac = sepp.sepp_full.Optimiser1Factory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=5)
model

fig = plot(model, space_size=1000, time_size=20, space_floor=np.exp(-20))

fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))

tk_time_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
tk_space_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(20)
opt_fac = sepp.sepp_full.Optimiser1SEMFactory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac) 
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=15)
model

fig = plot(model, space_size=750, time_size=200, space_floor=np.exp(-20))

fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))

backgrounds, trigger_deltas = trainer.sample_to_points(model, datetime.datetime(2017,1,1))
backgrounds.shape, trigger_deltas.shape

tk_time_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
tk_space_prov = sepp.kernels.NearestNeighbourKernelProvider(15)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(15)
opt_fac = sepp.sepp_full.Optimiser1Factory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=5)
model

fig = plot(model, space_size=800, time_size=15, space_floor=np.exp(-20), geo=northside, grid=grid)

fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))

for _ in range(10):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()

fig = plot(model, space_size=1300, time_size=15, space_floor=np.exp(-20), geo=northside, grid=grid)

fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))

tk_time_prov = sepp.kernels.NearestNeighbourKernelProvider(30)
tk_space_prov = sepp.kernels.NearestNeighbourKernelProvider(15)
back_ker_prov = sepp.kernels.NearestNeighbourKernelProvider(15)
opt_fac = sepp.sepp_full.Optimiser1SEMFactory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac) 
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=15)
model

fig = plot(model, space_size=1300, time_size=15, space_floor=np.exp(-20), geo=northside, grid=grid)

fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))



