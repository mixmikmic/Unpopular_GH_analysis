import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import sepp.sepp_grid
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

northside = open_cp.sources.chicago.get_side("Southwest")
grid = open_cp.data.Grid(150, 150, 0, 0)
grid = open_cp.geometry.mask_grid_by_intersection(northside, grid)

mask = (all_points.timestamps >= np.datetime64("2016-01-01")) & (all_points.timestamps < np.datetime64("2017-01-01"))
points = all_points[mask]
print(points.number_data_points)
points = open_cp.geometry.intersect_timed_points(points, northside)
print(points.number_data_points)

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

trainer = sepp.sepp_grid.ExpDecayTrainer(grid)
trainer.data = points
model = trainer.train(datetime.datetime(2017,1,1), iterations=50)

cells = trainer.to_cells(datetime.datetime(2017,1,1))

cells.shape

cells, model = trainer.initial_model(datetime.datetime(2017,1,1))
model

for _ in range(50):
    opt = sepp.sepp_grid.ExpDecayOptFast(model, cells)
    model = opt.optimised_model()
    print(model)

cells, model = trainer.initial_model(datetime.datetime(2017,1,1))
new_cells = []
for cell in cells.flat:
    if len(cell) > 0:
        cell += np.random.random(size=len(cell)) * 0.1
        cell.sort()
        assert all(x<0 for x in cell)
    new_cells.append(cell)
cells = np.asarray(new_cells).reshape(cells.shape)

for _ in range(50):
    opt = sepp.sepp_grid.ExpDecayOptFast(model, cells)
    model = opt.optimised_model()
    print(model)

import impute.chicago
import shapely.geometry

def gen():
    with lzma.open(os.path.join(datadir, "chicago_all.csv.xz"), "rt") as file:
        yield from impute.chicago.load_only_with_point(file)
next(gen())

proj = impute.chicago.projector()
rows = []
for row in gen():
    in_time_range = row.datetime >= datetime.datetime(2016,1,1) and row.datetime < datetime.datetime(2017,1,1)
    if in_time_range and row.crime_type=="BURGLARY":
        point = shapely.geometry.Point(*proj(*row.point))
        if northside.intersects(point):
            rows.append(row)
rows.sort(key = lambda row : row.datetime)
len(rows)

points.number_data_points

cells = np.empty((grid.yextent, grid.xextent), dtype=np.object)
for x in range(grid.xextent):
    for y in range(grid.yextent):
        cells[y, x] = list()
for row in rows:
    x, y = grid.grid_coord(*proj(*row.point))
    cells[y,x].append(row)

[x for x in cells.flat if len(x)>10]



