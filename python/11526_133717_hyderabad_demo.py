get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import datacube
from datacube.model import Range
from datetime import datetime
dc = datacube.Datacube(app='dc-example')
from datacube.storage import masking
from datacube.storage.masking import mask_valid_data as mask_invalid_data
import pandas
import xarray
import numpy
import json
import vega
from datacube.utils import geometry
numpy.seterr(divide='ignore', invalid='ignore')

import folium
from IPython.display import display
import geopandas
from shapely.geometry import mapping
from shapely.geometry import MultiPolygon
import rasterio
import shapely.geometry
import shapely.ops
from functools import partial
import pyproj
from datacube.model import CRS
from datacube.utils import geometry

## From http://scikit-image.org/docs/dev/auto_examples/plot_equalize.html
from skimage import data, img_as_float
from skimage import exposure

datacube.__version__

def datasets_union(dss):
    thing = geometry.unary_union(ds.extent for ds in dss)
    return thing.to_crs(geometry.CRS('EPSG:4326'))

import random
def plot_folium(shapes):

    mapa = folium.Map(location=[17.38,78.48], zoom_start=8)
    colors=['#00ff00', '#ff0000', '#00ffff', '#ffffff', '#000000', '#ff00ff']
    for shape in shapes:
        style_function = lambda x: {'fillColor': '#000000' if x['type'] == 'Polygon' else '#00ff00', 
                                   'color' : random.choice(colors)}
        poly = folium.features.GeoJson(mapping(shape), style_function=style_function)
        mapa.add_children(poly)
    display(mapa)

# determine the clip parameters for a target clear (cloud free image) - identified through the index provided
def get_p2_p98(rgb, red, green, blue, index):

    r = numpy.nan_to_num(numpy.array(rgb.data_vars[red][index]))
    g = numpy.nan_to_num(numpy.array(rgb.data_vars[green][index]))
    b = numpy.nan_to_num(numpy.array(rgb.data_vars[blue][index]))
  
    rp2, rp98 = numpy.percentile(r, (2, 99))
    gp2, gp98 = numpy.percentile(g, (2, 99)) 
    bp2, bp98 = numpy.percentile(b, (2, 99))

    return(rp2, rp98, gp2, gp98, bp2, bp98)

def plot_rgb(rgb, rp2, rp98, gp2, gp98, bp2, bp98, red, green, blue, index):

    r = numpy.nan_to_num(numpy.array(rgb.data_vars[red][index]))
    g = numpy.nan_to_num(numpy.array(rgb.data_vars[green][index]))
    b = numpy.nan_to_num(numpy.array(rgb.data_vars[blue][index]))

    r_rescale = exposure.rescale_intensity(r, in_range=(rp2, rp98))
    g_rescale = exposure.rescale_intensity(g, in_range=(gp2, gp98))
    b_rescale = exposure.rescale_intensity(b, in_range=(bp2, bp98))

    rgb_stack = numpy.dstack((r_rescale,g_rescale,b_rescale))
    img = img_as_float(rgb_stack)

    return(img)

def plot_water_pixel_drill(water_drill):
    vega_data = [{'x': str(ts), 'y': str(v)} for ts, v in zip(water_drill.time.values, water_drill.values)]
    vega_spec = """{"width":720,"height":90,"padding":{"top":10,"left":80,"bottom":60,"right":30},"data":[{"name":"wofs","values":[{"code":0,"class":"dry","display":"Dry","color":"#D99694","y_top":30,"y_bottom":50},{"code":1,"class":"nodata","display":"No Data","color":"#A0A0A0","y_top":60,"y_bottom":80},{"code":2,"class":"shadow","display":"Shadow","color":"#A0A0A0","y_top":60,"y_bottom":80},{"code":4,"class":"cloud","display":"Cloud","color":"#A0A0A0","y_top":60,"y_bottom":80},{"code":1,"class":"wet","display":"Wet","color":"#4F81BD","y_top":0,"y_bottom":20},{"code":3,"class":"snow","display":"Snow","color":"#4F81BD","y_top":0,"y_bottom":20},{"code":255,"class":"fill","display":"Fill","color":"#4F81BD","y_top":0,"y_bottom":20}]},{"name":"table","format":{"type":"json","parse":{"x":"date"}},"values":[],"transform":[{"type":"lookup","on":"wofs","onKey":"code","keys":["y"],"as":["class"],"default":null},{"type":"filter","test":"datum.y != 255"}]}],"scales":[{"name":"x","type":"time","range":"width","domain":{"data":"table","field":"x"},"round":true},{"name":"y","type":"ordinal","range":"height","domain":["water","not water","not observed"],"nice":true}],"axes":[{"type":"x","scale":"x","formatType":"time"},{"type":"y","scale":"y","tickSize":0}],"marks":[{"description":"data plot","type":"rect","from":{"data":"table"},"properties":{"enter":{"xc":{"scale":"x","field":"x"},"width":{"value":"1"},"y":{"field":"class.y_top"},"y2":{"field":"class.y_bottom"},"fill":{"field":"class.color"},"strokeOpacity":{"value":"0"}}}}]}"""
    spec_obj = json.loads(vega_spec)
    spec_obj['data'][1]['values'] = vega_data
    return vega.Vega(spec_obj)

plot_folium([datasets_union(dc.index.datasets.search_eager(product='ls5_ledaps_scene')),             datasets_union(dc.index.datasets.search_eager(product='ls7_ledaps_scene')),             datasets_union(dc.index.datasets.search_eager(product='ls8_ledaps_scene'))])

dc.list_measurements()

# Hyderbad
#    'lon': (78.40, 78.57),
#    'lat': (17.36, 17.52),
# Lake Singur
#    'lat': (17.67, 17.84),
#    'lon': (77.83, 78.0),

# Lake Singur Dam
query = {
    'lat': (17.72, 17.79),
    'lon': (77.88, 77.95),
}

products = ['ls5_ledaps_scene','ls7_ledaps_scene','ls8_ledaps_scene']

datasets = []
for product in products:
    ds = dc.load(product=product, measurements=['nir','red', 'green','blue'], output_crs='EPSG:32644',resolution=(-30,30), **query)
    ds['product'] = ('time', numpy.repeat(product, ds.time.size))
    datasets.append(ds)

sr = xarray.concat(datasets, dim='time')
sr = sr.isel(time=sr.time.argsort())  # sort along time dim
sr = sr.where(sr != -9999)

##### include an index here for the timeslice with representative data for best stretch of time series

# don't run this to keep the same limits as the previous sensor
#rp2, rp98, gp2, gp98, bp2, bp98 = get_p2_p98(sr,'red','green','blue', 0)

rp2, rp98, gp2, gp98, bp2, bp98 = (300.0, 2000.0, 300.0, 2000.0, 300.0, 2000.0)
print(rp2, rp98, gp2, gp98, bp2, bp98)

plt.imshow(plot_rgb(sr,rp2, rp98, gp2, gp98, bp2, bp98,'red',
                        'green', 'blue', 0),interpolation='nearest')

datasets = []
for product in products:
    ds = dc.load(product=product, measurements=['cfmask'], output_crs='EPSG:32644',resolution=(-30,30), **query).cfmask
    ds['product'] = ('time', numpy.repeat(product, ds.time.size))
    datasets.append(ds)

pq = xarray.concat(datasets, dim='time')
pq = pq.isel(time=pq.time.argsort())  # sort along time dim
del(datasets)

pq.attrs['flags_definition'] = {'cfmask': {'values': {'255': 'fill', '1': 'water', '2': 'shadow', '3': 'snow', '4': 'cloud', '0': 'clear'}, 'description': 'CFmask', 'bits': [0, 1, 2, 3, 4, 5, 6, 7]}}

pandas.DataFrame.from_dict(masking.get_flags_def(pq), orient='index')

water = masking.make_mask(pq, cfmask ='water')
water.sum('time').plot(cmap='nipy_spectral')

plot_water_pixel_drill(pq.isel(y=int(water.shape[1] / 2), x=int(water.shape[2] / 2)))

del(water)

mask = masking.make_mask(pq, cfmask ='cloud')
mask = abs(mask*-1+1)
sr = sr.where(mask)
mask = masking.make_mask(pq, cfmask ='shadow')
mask = abs(mask*-1+1)
sr = sr.where(mask)
del(mask)
del(pq)

sr.attrs['crs'] = CRS('EPSG:32644')

ndvi_median = ((sr.nir-sr.red)/(sr.nir+sr.red)).median(dim='time')
ndvi_median.attrs['crs'] = CRS('EPSG:32644')
ndvi_median.plot(cmap='YlGn', robust='True')

poi_latitude = 17.749343
poi_longitude = 77.935634

p = geometry.point(x=poi_longitude, y=poi_latitude, crs=geometry.CRS('EPSG:4326')).to_crs(sr.crs)

subset = sr.sel(x=((sr.x > p.points[0][0]-1000)), y=((sr.y < p.points[0][1]+1000)))
subset = subset.sel(x=((subset.x < p.points[0][0]+1000)), y=((subset.y > p.points[0][1]-1000)))

plt.imshow(plot_rgb(subset,rp2, rp98, gp2, gp98, bp2, bp98,'red',
                        'green', 'blue',0),interpolation='nearest' )

((sr.nir-sr.red)/(sr.nir+sr.red)).sel(x=p.points[0][0], y=p.points[0][1], method='nearest').plot(marker='o')

