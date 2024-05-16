get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import mmmpy

path = '/Users/tjlang/Documents/Python/DataForTesting/'
tile1a = mmmpy.MosaicTile(path + '20130601/mosaic3d_tile1_20130601-020000.netcdf', 
                          verbose=True)
tile2a = mmmpy.MosaicTile(path + '20130601/mosaic3d_tile2_20130601-020000.netcdf')
tile3a = mmmpy.MosaicTile(path + '20130601/mosaic3d_tile3_20130601-020000.netcdf')
tile4a = mmmpy.MosaicTile(path + '20130601/mosaic3d_tile4_20130601-020000.netcdf')
tile5a = mmmpy.MosaicTile(path + '20130601/mosaic3d_tile5_20130601-020000.netcdf')
tile6a = mmmpy.MosaicTile(path + '20130601/mosaic3d_tile6_20130601-020000.netcdf')
tile7a = mmmpy.MosaicTile(path + '20130601/mosaic3d_tile7_20130601-020000.netcdf')
tile8a = mmmpy.MosaicTile(path + '20130601/mosaic3d_tile8_20130601-020000.netcdf')

tile6a.diag(verbose=True)

dir(tile6a)

print(tile6a.Filename)
print(tile6a.Version)
print(tile6a.Variables)
print(tile6a.nlat, tile6a.nlon, tile6a.nz)
print(tile6a.StartLat, tile6a.StartLon, tile6a.LatGridSpacing, tile6a.LonGridSpacing)
print(tile6a.Time, tile6a.Duration)

display = mmmpy.MosaicDisplay(tile6a)
help(display.plot_horiz)

fig = plt.figure(figsize=(8,8))
display.plot_horiz(level=4.1, verbose=True, latrange=[40,20], lonrange=[-90,-110], 
                   save='tile6.png', clevs=4.0*np.arange(20), cmap='CMRmap', show_grid=False, 
                   title='Tile 6')

help(display.plot_vert)

display.plot_vert(lat=36.5, xrange=[-99,-93], cmap='rainbow')

display.plot_vert(lon=-97.1, xrange=[35,36], cmap='rainbow', zrange=[0,15], 
                  title='Longitude = 97.1 W')

display.three_panel_plot(lat=35.25, lon=-97.1, latrange=[33,39], lonrange=[-99,-93], level=4,
                         meridians=1, parallels=1, xrange_b=[-98.5, -96.5], xrange_c=[34.5, 36.5])

map_array_a = [ [tile1a, tile2a, tile3a, tile4a], [tile5a, tile6a, tile7a, tile8a] ]
stitch_a = mmmpy.stitch_mosaic_tiles(map_array=map_array_a, verbose=True)

help(mmmpy.stitch_mosaic_tiles)

display_a = mmmpy.MosaicDisplay(stitch_a)
display_a.plot_horiz(parallels=5)

fig = plt.figure(figsize=(6,6))
display_a.plot_horiz(level=4, latrange=[30, 45], lonrange=[-85, -100], 
                     show_grid=False, title='Stitched at Last!')

stitch_small = mmmpy.stitch_mosaic_tiles(map_array=[tile6a, tile7a], direction='we')
small_display = mmmpy.MosaicDisplay(stitch_small)
small_display.plot_horiz(latrange=[np.min(stitch_small.Latitude),
                                   np.max(stitch_small.Latitude)],
                         lonrange=[np.min(stitch_small.Longitude),
                                   np.max(stitch_small.Longitude)],
                         meridians=5, parallels=5)

fig = plt.figure(figsize=(8, 8))
stitch_small = mmmpy.stitch_mosaic_tiles(map_array=[[tile2a, tile3a], [tile6a, tile7a]])
small_display = mmmpy.MosaicDisplay(stitch_small)
small_display.plot_horiz(latrange=[np.min(stitch_small.Latitude),
                                   np.max(stitch_small.Latitude)],
                         lonrange=[np.min(stitch_small.Longitude),
                                   np.max(stitch_small.Longitude)],
                         meridians=5, parallels=5)

tile1b = mmmpy.MosaicTile(path + './V2_MRMS_Data/tile1/20140304-140000.netcdf')
tile2b = mmmpy.MosaicTile(path + './V2_MRMS_Data/tile2/20140304-140000.netcdf')
tile3b = mmmpy.MosaicTile(path + './V2_MRMS_Data/tile3/20140304-140000.netcdf')
tile4b = mmmpy.MosaicTile(path + './V2_MRMS_Data/tile4/20140304-140000.netcdf')
map_array_b = [[tile1b, tile2b], [tile3b, tile4b]]
stitch_b = mmmpy.stitch_mosaic_tiles(map_array_b)
display = mmmpy.MosaicDisplay(stitch_b)
display.plot_horiz(level=4)

print(tile1b.Filename)
print(tile1b.Version)
print(tile1b.Variables)
print(tile1b.nlat, tile1b.nlon, tile1b.nz)
print(tile1b.StartLat, tile1b.StartLon, tile1b.LatGridSpacing, tile1b.LonGridSpacing)
print(tile1b.Time, tile1b.Duration)

file_dir = '/Users/tjlang/Documents/Python/DataForTesting/binary/'
tile2 = mmmpy.MosaicTile(file_dir+'MREF3D33L_tile2.20140619.010000.gz', verbose=True)
d2 = mmmpy.MosaicDisplay(tile2)
d2.three_panel_plot(lat=40.75, lon=-80.75, latrange=[38, 45], 
                    lonrange=[-86, -77], meridians=2, parallels=3, xrange_b=[-83, -79], 
                    xrange_c=[43, 40], show_grid=True, zrange=[0,15], level=4)

help(mmmpy)



