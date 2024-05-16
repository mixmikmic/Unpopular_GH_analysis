import pandas as pd

frame = pd.read_csv("example.csv")
frame.head()

import dateutil.parser

timestamps = frame.Date.map(dateutil.parser.parse)
timestamps[:5]

import pyproj
proj = pyproj.Proj({"init":"EPSG:2790"})

xcoords, ycoords = proj(frame.Longitude.values, frame.Latitude.values)
xcoords[:5], ycoords[:5]

import open_cp.data

points = open_cp.data.TimedPoints.from_coords(timestamps, xcoords, ycoords)

points.number_data_points

points.time_range

import geopandas as gpd

frame = gpd.read_file("SouthSide")
frame

geo = list(frame.geometry)[0]

import descartes
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect(1)
ax.add_patch(descartes.PolygonPatch(geo, fc="none"))
ax.scatter(xcoords, ycoords, marker="x", color="black", linewidth=1)

import open_cp.data
import open_cp.geometry

grid = open_cp.data.Grid(xsize=150, ysize=150, xoffset=0, yoffset=0)
grid = open_cp.geometry.mask_grid_by_intersection(geo, grid)

import open_cp.plot
import matplotlib.collections

fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect(1)
lc = matplotlib.collections.LineCollection(open_cp.plot.lines_from_grid(grid), color="black", linewidth=1)
ax.add_collection(lc)
ax.scatter(xcoords, ycoords, marker="x", color="black", linewidth=0.5)

import sepp.sepp_grid
import datetime

trainer = sepp.sepp_grid.ExpDecayTrainer(grid)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1))

model

fig, ax = plt.subplots(figsize=(10,6))

ax.add_patch(descartes.PolygonPatch(geo, fc="none"))
ax.set_aspect(1)
pred = trainer.prediction_from_background(model)
mappable = ax.pcolor(*pred.mesh_data(), pred.intensity_matrix, cmap="Greys", rasterized=True)
fig.colorbar(mappable, ax=ax)
ax.set_title("Estimated background rate")
None

import sepp.kernels
import sepp.sepp_full
import numpy as np
import scipy.stats

tk_time_prov = sepp.kernels.FixedBandwidthKernelProvider(1)
tk_space_prov = sepp.kernels.FixedBandwidthKernelProvider(20, cutoff=750)
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(100)
opt_fac = sepp.sepp_full.Optimiser1Factory(back_ker_prov, tk_time_prov, tk_space_prov)
trainer = sepp.sepp_full.Trainer1(opt_fac)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=5)
model

def compute_plot_kde(ker, size):
    x = np.linspace(-size, size, 151)
    y = x
    xcs, ycs = np.meshgrid(x, y)
    z = ker([xcs.flatten(), ycs.flatten()])
    z = z.reshape(xcs.shape)
    return x, y, z

def plot_kde(ax, ker, size, postprocess=None):
    x, y, z = compute_plot_kde(ker, size)
    if postprocess is not None:
        z = postprocess(z)
    return ax.pcolormesh(x,y,z, cmap="Greys", rasterized=True)

def backup_limits(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    return xmin, xmax, ymin, ymax

def set_limits(ax, xmin, xmax, ymin, ymax):
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    
def plot(model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(geo, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * model.trigger_time_kernel(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], model.trigger_space_kernel, space_size, pp)
    plt.colorbar(m, ax=axes[2])
        
    fig.tight_layout()
    return fig

def plot_scatter_triggers(backgrounds, trigger_deltas):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    def add_kde(ax, pts):
        xmin, xmax, ymin, ymax = backup_limits(ax)
        x = np.linspace(xmin, xmax, 151)
        y = np.linspace(ymin, ymax, 151)
        xcs, ycs = np.meshgrid(x, y)
        ker = scipy.stats.kde.gaussian_kde(pts)
        z = ker([xcs.flatten(), ycs.flatten()])
        z = z.reshape(xcs.shape)
        z = np.log(np.exp(-15)+z)
        m = ax.pcolorfast(x,y,z, cmap="Greys", rasterized=True, alpha=0.7, zorder=-10)

    ax = axes[0]
    pts = trigger_deltas[1:]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set_title("Space trigger points")

    ax = axes[1]
    pts = trigger_deltas[[0,1]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="x coord")#, xlim=[0,200])

    ax = axes[2]
    pts = trigger_deltas[[0,2]]
    ax.scatter(*pts, marker="x", color="black", linewidth=1)
    add_kde(ax, pts)
    ax.set(xlabel="days", ylabel="y coord")

    fig.tight_layout()
    return fig

def scatter_triggers(trainer, model, predict_time):
    backgrounds, trigger_deltas = trainer.sample_to_points(model, predict_time)
    return plot_scatter_triggers(backgrounds, trigger_deltas), backgrounds, trigger_deltas

fig = plot(model, space_size=750, time_size=20, space_floor=np.exp(-20))

fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))



