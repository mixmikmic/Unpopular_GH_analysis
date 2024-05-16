get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os, lzma, csv, bz2
import tilemapbase
import numpy as np

#datadir = os.path.join("/media", "disk", "Data")
datadir = os.path.join("..", "..", "..", "..", "Data")

import opencrimedata.chicago as chicago
import opencrimedata
print(opencrimedata.__version__)

filename = os.path.join(datadir, "chicago_all.csv.xz")
def gen():
    with lzma.open(filename, "rt", encoding="UTF8") as f:
        yield from chicago.load_only_with_point(f)
        
next(gen())

coords_wm = np.asarray([tilemapbase.project(*row.point) for row in gen() if row.crime_type=="BURGLARY"])

def gen_new():
    fn = os.path.join(datadir, "chicago_redist_network_flow_to_buildings_network.csv.xz")
    with lzma.open(fn, "rt", encoding="UTF8") as f:
        yield from chicago.load_only_with_point(f)
        
coords_new_wm = np.asarray([tilemapbase.project(*row.point) for row in gen_new() if row.crime_type=="BURGLARY"])

fig, axes = plt.subplots(ncols=2, figsize=(17,8))

ex = tilemapbase.Extent.from_centre(0.25662, 0.3722, xsize=0.00005)
plotter = tilemapbase.Plotter(ex, tilemapbase.tiles.OSM, width=800)
for ax in axes:
    plotter.plot(ax)
axes[0].scatter(*coords_wm.T, marker="x", color="black", alpha=0.5)
axes[1].scatter(*coords_new_wm.T, marker="x", color="black", alpha=0.5)
None

fig.savefig("Chicago_overview.png", dpi=150)

import open_cp.sources.chicago as chicago
chicago.set_data_directory(datadir)

chicago.get_side("South")



