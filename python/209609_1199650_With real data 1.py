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
import pickle, lzma, datetime, collections
import open_cp.predictors
import scipy.stats
import sepp.kernels

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
    
def plot(trainer, data, model, space_size=35, time_size=100, space_floor=None):
    fig, axes = plt.subplots(ncols=3, figsize=(16,5))

    ax = axes[0]
    ax.add_patch(descartes.PolygonPatch(northside, fc="none"))
    ax.set_aspect(1)
    bpred = open_cp.predictors.grid_prediction_from_kernel(model.background_kernel, grid.region(), grid.xsize)
    #bpred = open_cp.predictors.GridPredictionArray(grid.xsize, grid.ysize, model.mu, grid.xoffset, grid.yoffset)
    m = ax.pcolor(*bpred.mesh_data(), bpred.intensity_matrix, cmap="Greys", rasterized=True)
    cb = fig.colorbar(m, ax=ax)

    # Quickly compute marginals...
    opt = trainer._optimiser(model, data)
    x, w = opt.data_for_trigger_opt()
    prov = opt_fac._trigger_provider
    ker = open_cp.kernels.GaussianBase(x)
    ker.covariance_matrix = np.diag(prov._scale)
    ker.bandwidth = prov._h
    ker.weights = w
    xy_marginal = open_cp.kernels.marginalise_gaussian_kernel(ker, axis=0)
    tx_marginal = open_cp.kernels.marginalise_gaussian_kernel(ker, axis=2)
    t_marginal = open_cp.kernels.marginalise_gaussian_kernel(tx_marginal, axis=1)
    
    ax = axes[1]
    x = np.linspace(0, time_size, 200)
    y = model.theta * t_marginal(x)
    ax.plot(x, y, color="black")
    ax.set(xlabel="Days", ylabel="Trigger risk")
    y = np.max(y)
    for t in range(0, time_size+1):
        ax.plot([t,t],[0,y], color="grey", linewidth=0.5, linestyle="--", zorder=-10)

    pp = None
    if space_floor is not None:
        pp = lambda z : np.log(space_floor + z)
    m = plot_kde(axes[2], xy_marginal, space_size, pp)
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

tk_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(1, scale=[1,20,20])
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(20)
opt_fac = sepp.sepp_full.OptimiserFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2017,1,1))
model = trainer.train(datetime.datetime(2017,1,1), iterations=20)

fig = plot(trainer, data, model, space_size=750, time_size=200, space_floor=np.exp(-20))

fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))

for _ in range(30):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model

fig = plot(trainer, data, model, space_size=750, time_size=200, space_floor=np.exp(-20))

fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))

tk_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(20, scale=[0.01,1,1])
back_ker_prov = sepp.kernels.FixedBandwidthKernelProvider(20)
opt_fac = sepp.sepp_full.OptimiserSEMFactory(back_ker_prov, tk_ker_prov)
trainer = sepp.sepp_full.Trainer(opt_fac, p_cutoff=99.99, initial_space_scale=100)
trainer.data = points
T, data = trainer.make_data(datetime.datetime(2018,1,1))
model = trainer.train(datetime.datetime(2018,1,1), iterations=20)

fig = plot(trainer, data, model, space_size=750, time_size=200, space_floor=np.exp(-20))

fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))

for _ in range(50):
    opt = trainer._optimiser(model, data)
    model = opt.iterate()
model

fig = plot(trainer, data, model, space_size=750, time_size=200, space_floor=np.exp(-20))

fig, *_ = scatter_triggers(trainer, model, datetime.datetime(2017,1,1))



