# io
import zipfile
import io
import bisect
import logging

# Import packages
import numpy as np
import pandas as pd

# download, formats and projections
import netCDF4
import pyproj
import requests
import geojson
import rtree
import owslib.wcs
import osgeo.osr
import rasterio

# rendering
import mako.template
import IPython.display

# plotting
import bokeh.models
import bokeh.tile_providers
import bokeh.plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.colors

get_ipython().magic('matplotlib inline')
bokeh.plotting.output_notebook()

# the files needed for this analysis
# dino, information about the subsoil >= -50m
dino_url = 'http://dinodata.nl/opendap/GeoTOP/geotop.nc'
# ddl, the data distributie laag
ddl_url = 'https://waterwebservices.rijkswaterstaat.nl/METADATASERVICES_DBO/OphalenCatalogus/'
# Dutch ordnance system
nap_url = 'https://geodata.nationaalgeoregister.nl/napinfo/wfs'
# Web coverage service for NAP.
ahn_url = 'http://geodata.nationaalgeoregister.nl/ahn2/wcs'
# The data from psmsl.org
psmsl_url = 'http://www.psmsl.org/data/obtaining/rlr.annual.data/rlr_annual.zip'

# These are not used.
# Using osgeo.osr is an alternative to pyproj.
# They can give different results if EPSG tables are not up to date
# or if the proj version does not support datum transformations (>4.6.0)
wgs84 = osgeo.osr.SpatialReference()
wgs84.ImportFromEPSG(4326)
rd = osgeo.osr.SpatialReference()
rd.ImportFromEPSG(28992)
rdnap = osgeo.osr.SpatialReference()
rdnap.ImportFromEPSG(7415)
webm = osgeo.osr.SpatialReference()
webm.ImportFromEPSG(3857)
etrs89_utm31n = osgeo.osr.SpatialReference()
etrs89_utm31n.ImportFromEPSG(25831)
etrs89_utm31n2rd = osgeo.osr.CoordinateTransformation(etrs89_utm31n, rd)
etrs89_utm31n2wgs84 = osgeo.osr.CoordinateTransformation(etrs89_utm31n, wgs84)
etrs89_utm31n2webm = osgeo.osr.CoordinateTransformation(etrs89_utm31n, webm)

# We use pyproj
wgs84 = pyproj.Proj(init='epsg:4326')
rd = pyproj.Proj(init='epsg:28992')
rdnap = pyproj.Proj(init='epsg:7415')
webm = pyproj.Proj(init='epsg:3857')
etrs89_utm31n = pyproj.Proj(init='epsg:25831')

# if pyproj is old, give an error
if str(wgs84.proj_version) < "4.60":
    logging.error("""pyproj version {} does not support datum transformations. Use osgeo.osr, example above.""".format(wgs84.proj_version))

# this is information collected manual

main_stations = [
    {
        "ddl_id": "DELFZL",
        "location": "Delfzijl",
        "psmsl_id": 24,
        "foundation_low": -20,
        "station_low": 1.85,
        "station_high": 10.18,
        "nulpaal": 0,
        "rlr2nap": lambda x: x - (6978-155),
        "summary": "The tidal measurement station in Delfzijl is located in the harbour of Delfzijl. Category 'Peilmeetstation'. The station has a main building with a foundation on a steel round pole (inner width = 2.3m, outer width 2.348m) reaching to a depth of -20m NAP. The Building is placed in the harbour and is connected to the main land by means of a steel stairs towards a quay. Which has also a foundation on steel poles. Peilbout is inside the construction attached to the wall. Every ten minutes the water level relative to NAP is measured. Between -4 and -5 m depth concrete seals of the underwater chamber. ",
        "img": "http://www.openearth.nl/sealevel/static/images/DELFZL.jpg",
        "autocad": "http://a360.co/2s8ltK7",      
        "links": []
    },
    {
        "ddl_id": "DENHDR",
        "location": "Den Helder",
        "psmsl_id": 23,
        "foundation_low": -5,
        "station_low": 5,
        "station_high": 8.47,
        "nulpaal": 1,
        "rlr2nap": lambda x: x - (6988-42),
        "summary": "This station is located in the dike of Den Helder. The station has a pipe through the dike towards the sea for the measurement of the water level. The inlet of this pipe is at -3.25m NAP. There is a seperate construction for the ventilation of the main building. Furthermore the peilbout is located outside the main building at the opposite side of the dike. The main construction has a foundation of steel sheet pilings forming a rectangle around the measurement instruments. Between -4 and -5 m depth concrete seals of the underwater chamber. ",
        "img": "http://www.openearth.nl/sealevel/static/images/DENHDR.jpg",
        "autocad": "http://a360.co/2sYyitj",
        "links": []
    },
    {
        "ddl_id": "HARLGN", 
        "location": "Harlingen",
        "psmsl_id": 25,
        "foundation_low": -5.4,
        "station_low": 5.55,
        "station_high": 8.54,
        "nulpaal": 1,
        "rlr2nap": lambda x: x - (7036-122),
        "summary": "The tidal station in Harlingen is located in a harbour on top of a boulevard. A pipe is going from the station at a depth of -2.56m NAP towards the sea. The inlet of the pipe is protected by a construction, so as to reduce the variations by the wave impact. The Main building has a foundation of a steel sheet pilings construction (rectangle inner dimensions 2.53 by 2.27m^2) surrounding the measurement instruments.",
        "img": "http://www.openearth.nl/sealevel/static/images/DELFZL.jpg",
        "autocad": "http://a360.co/2sYfFFX",
        "links": []
    },
    {
        "ddl_id": "HOEKVHLD", 
        "location": "Hoek van Holland",
        "psmsl_id": 22,
        "foundation_low": -3.3,
        "station_low": 5.27,
        "station_high": 9.05,
        "nulpaal": 0,
        "rlr2nap": lambda x:x - (6994 - 121),
        "summary": "The station in Hoek van Holland is located beside the Nieuwe Waterweg near the river mouth into the North Sea. The reference pole is situated outside the main building on the main land. The main building is connected to the main land by a steel bridge. The foundation of the main building is on steel poles. The building is a concrete structure reaching to a depth of -3.0 m NAP. This entire thing is enough for the measurement instruments to be placed inside. And the underwater chamber is then, in contrary to the other stations within the main building. The entire concrete structure has a foundation of multiple sheet piles. These are 8 concrete plates (8-sided) with a length of 14.1m. ",
        "img": "http://www.openearth.nl/sealevel/static/images/HOEKVHLD.jpg",
        "autocad": "http://a360.co/2uqAgAs",
        "links": []
    },
    {
        "ddl_id": "IJMDBTHVN", 
        "location": "IJmuiden",
        "psmsl_id": 32,
        "foundation_low": -13,
        "station_low": 4.2,
        "station_high": 10.35,
        "nulpaal": 0,
        "rlr2nap": lambda x: x - (7033-83),
        "summary": "IJmuiden is located on the northern part of the marina in IJmuiden, near a breakwater. The main building is situated in the water and is connected by a steel stairs and bridge with the main land. The foundation of this building consists out a round steel sheet pile. The under water chamber is closed of with a concrete slab between -3.75m NAP and - 4.5m NAP. The sheet pile is extended to a depth of -13m NAP. IJmuiden has a GPS (GNSS) station attached to it.",
        "img": "http://www.openearth.nl/sealevel/static/images/IJMDBTHVN.jpg",
        "autocad": "http://a360.co/2sZ4Nrn",
        "links": [
            {
                "href": "http://gnss1.tudelft.nl/dpga/station/Ijmuiden.html",
                "name": "GNSS info"
            }
        ]
    },
    {
        "ddl_id": "VLISSGN",
        "location": "Vlissingen",
        "psmsl_id": 20,
        "foundation_low": -17.6,
        "station_low": 2.5,
        "station_high": 9,
        "nulpaal": 0,
        "rlr2nap": lambda x: x - (6976-46),
        "summary": "This station is located at a quay in Vlissingen, near the outer harbour. The foundation is a steel sheet pile reaching to a depth of -17.6m NAP, having a width of 2.2m (outer width). Inside this pile are the measurement instruments. The under water chamber is sealed of with a concrete slab reaching from -4.0 m NAP to -5.0 m NAP. The station has a GPS (GNSS) device attached.",
        "img": "http://www.openearth.nl/sealevel/static/images/VLISSGN.jpg",
        "autocad": "http://a360.co/2sZ4Nrn",
        "links": [
            {
                "href": "http://gnss1.tudelft.nl/dpga/station/Vlissingen.html",
                "name": "GNSS info"
            }
        ]
    }
]

# convert the data to a dataframe (table)
station_data = pd.DataFrame.from_records(main_stations)
station_data = station_data.set_index('ddl_id')
station_data[['location', 'psmsl_id', 'nulpaal']]

# read information from the psmsl
zf = zipfile.ZipFile('../data/psmsl/rlr_annual.zip')
records = []
for station in main_stations:
    filename = 'rlr_annual/RLR_info/{}.txt'.format(station['psmsl_id'])
    img_bytes = zf.read('rlr_annual/RLR_info/{}.png'.format(station['psmsl_id']))
    img = plt.imread(io.BytesIO(img_bytes))
    record = {
        "ddl_id": station["ddl_id"],
        "psmsl_info": zf.read(filename).decode(),
        "psmsl_img": img
    }
    records.append(record)
psmsl_df = pd.DataFrame.from_records(records).set_index("ddl_id")
station_data = pd.merge(station_data, psmsl_df, left_index=True, right_index=True)
station_data[['psmsl_info']]

# read the grid of the dino dataset
dino = netCDF4.Dataset(dino_url, 'r')
x_dino = dino.variables['x'][:]
y_dino = dino.variables['y'][:]
z_dino = dino.variables['z'][:]
# lookup z index of -15m
z_min_idx = np.searchsorted(z_dino, -15)
# lookup litho at z index
z_min = dino.variables['lithok'][..., z_min_idx]
# keep the mask so we can look for close points

# fill value is sometimes a string, not sure why
fill_value = int(dino.variables['lithok']._FillValue)
mask_dino = np.ma.masked_equal(z_min, fill_value).mask

# get station information from DDL
request = {
    "CatalogusFilter": {
        "Eenheden": True,
        "Grootheden": True,
        "Hoedanigheden": True
    }
}
resp = requests.post(ddl_url, json=request)
result = resp.json()

df = pd.DataFrame.from_dict(result['LocatieLijst'])
df = df.set_index('Code')
# note that there are two stations for IJmuiden. 
# The station was moved from the sluices to outside of the harbor in 1981.
ids = ['DELFZL', 'DENHDR', 'HARLGN', 'HOEKVHLD', 'IJMDBTHVN', 'IJMDNDSS', 'VLISSGN']

# make a copy so we can add things
stations_df = df.loc[ids].copy()
# this drops IJMDNSS
stations_df = pd.merge(stations_df, station_data, left_index=True, right_index=True)
stations_df[['Naam', 'X', 'Y']]

# compute coordinates in different coordinate systems
stations_df['x_rd'], stations_df['y_rd'] = pyproj.transform(
    etrs89_utm31n, 
    rd, 
    list(stations_df.X), 
    list(stations_df.Y)
)
stations_df['lon'], stations_df['lat'] = pyproj.transform(
    etrs89_utm31n, 
    wgs84, 
    list(stations_df.X), 
    list(stations_df.Y)
)
stations_df['x_webm'], stations_df['y_webm'] = pyproj.transform(
    etrs89_utm31n, 
    webm, 
    list(stations_df.X), 
    list(stations_df.Y)
)
stations_df[['lat', 'lon']]

# We define some colors so they look somewhat natural. 
# The colors are based on images of the corresponding soil type.
# It is unknown what litho class 4 is. 

colors = {
    0: '#669966', # Above ground
    1: '#845F4C', # Peat
    2: '#734222', # Clay
    3: '#B99F71', # Sandy Clay
    4: '#ff0000', # litho 4
    5: '#E7D1C1', # Fines 
    6: '#c2b280', # Intermediate
    7: '#969CAA', # Coarse
    8: '#D0D6D6', # Gravel
    9: '#E5E5DB', # Shells,
    10: '#EEEEEE' # Undefined
    
}
labels = {
    0: "Above ground",
    1: "Peat",
    2: "Clay",
    3: "Sandy clay",
    4: "lithoclass 4",
    5: "Fine sand",
    6: "Intermediate fine sand",
    7: "Coarse sand",
    8: "Gravel",
    9: "Shells",
    10: "Undefined"
}

# Create a map of tide gauges with lithography at deep level
colors_rgb = [
    matplotlib.colors.hex2color(val) 
    for val 
    in colors.values()
]
# lookup colors
img = np.take(colors_rgb, np.ma.masked_values(z_min, -127).filled(10).T, axis=0)
fig, ax = plt.subplots(figsize=(8, 8/1.4))
ax.imshow(
    img, 
    origin='bottom', 
    extent=(x_dino[0], x_dino[-1], y_dino[0], y_dino[-1])
)
ax.plot(stations_df.x_rd, stations_df.y_rd, 'r.')
for name, row in stations_df.iterrows():
    ax.text(row.x_rd, row.y_rd, row.location, horizontalalignment='right')
_ = ax.set_title('Tide gauges with lithography at {}m'.format(z_dino[z_min_idx]))

# this part looks up the nearest location of DINO and looksup the lithography in that location

Y_dino, X_dino = np.meshgrid(y_dino, x_dino)
# Lookup the closest points in the dino database

# closest location
dino_idx = []
lithos = []
for code, station in stations_df.iterrows():
    # compute the distance
    x_idx = np.argmin(np.abs(station.x_rd - x_dino))
    y_idx = np.argmin(np.abs(station.y_rd - y_dino))
    # closest point, can also use a kdtree
    # store it
    dino_idx.append((x_idx, y_idx))
    lithok = dino.variables['lithok'][x_idx, y_idx, :]
    litho = pd.DataFrame(data=dict(z=z_dino, litho=lithok))
    lithos.append(litho)

# convert to array
dino_idx = np.array(dino_idx)
# store the tuples
stations_df['dino_idx'] = list(dino_idx)
# lookup x,y
stations_df['x_dino'] = x_dino[dino_idx[:, 0]]
stations_df['y_dino'] = y_dino[dino_idx[:, 1]]
stations_df['x_dino_webm'], stations_df['y_dino_webm'] = pyproj.transform(
    rd, 
    webm, 
    list(stations_df['x_dino']), 
    list(stations_df['y_dino'])
)
stations_df['lithok'] = lithos
stations_df[['lithok', 'x_dino', 'y_dino']]


features = geojson.load(open('../data/rws/nap/public/napinfo.json'))
index = rtree.Rtree()
for i, feature in enumerate(features['features']):
    # broken element (invalid coordinates)
    if feature['properties']['gml_id'] == "nappeilmerken.37004":
        continue
    index.add(i, tuple(feature['geometry']['coordinates']), feature)

# index.nearest(statio)
records = []
for ddl_id, station in stations_df.iterrows():
    closest = []
    for item in index.nearest((station.x_rd, station.y_rd), num_results=5, objects=True):
        feature = item.object
        feature['properties']['x_webm'], feature['properties']['y_webm'] = pyproj.transform(
            rd, 
            webm, 
            feature['properties']['x_rd'],
            feature['properties']['y_rd']
        )
        feature['properties']['distance'] = np.sqrt(
            (station.x_rd - float(feature['properties']['x_rd']))**2 +
            (station.y_rd - float(feature['properties']['y_rd']))**2
        )
            
        closest.append(feature)
    records.append({
        "ddl_id": ddl_id,
        "nap": closest
    })
nap_df = pd.DataFrame.from_records(records).set_index('ddl_id')

stations_df = pd.merge(stations_df, nap_df, left_index=True, right_index=True)
stations_df[['nap']]

wcs = owslib.wcs.WebCoverageService(ahn_url, version='1.0.0')

def ahn_for_station(station):
    # can't import this before pyproj are set
    

    # wms.getfeatureinfo()
    delta = 1e-6
    resp = wcs.getCoverage(
        'ahn2:ahn2_05m_int', 
        bbox=(station['lon']-delta, station['lat']-delta, station['lon']+delta, station['lat']+delta),
        crs='EPSG:4326',
        width=1,
        height=1,
        format='geotiff'
    )
    with open('result.tiff', 'wb') as f:
        f.write(resp.read())
    with rasterio.open('result.tiff') as f:
        data = f.read()[0]
    ahn_ma = np.ma.masked_outside(data, -100, 100)
    return ahn_ma[0, 0]

ahns = []
for ddl_id, station in stations_df.iterrows():
    ahn = ahn_for_station(station)
    ahns.append({
        "ddl_id": ddl_id,
        "ahn": ahn
    })
ahn_df = pd.DataFrame.from_records(ahns).set_index('ddl_id')
stations_df = pd.merge(stations_df, ahn_df, left_index=True, right_index=True)
stations_df[['ahn']]

# Now we plot the mapping between the DINO, NAP locations and the tide gauge locations

# Save data in datasource
stations_cds = bokeh.models.ColumnDataSource.from_df(
    stations_df[['x_webm', 'y_webm', 'x_dino_webm', 'y_dino_webm', 'Naam', 'lat', 'lon']].copy()
)

# Plot map with locations of stations and nearest data of dino_loket
p = bokeh.plotting.figure(tools='pan, wheel_zoom, box_zoom', x_range=(320000, 780000), y_range=(6800000, 7000000))

p.axis.visible = False
p.add_tile(bokeh.tile_providers.CARTODBPOSITRON)
# two layers
c1 = p.circle(x='x_webm', y='y_webm', size=20, source=stations_cds, legend='Tide gauges')
c2 = p.circle(x='x_dino_webm', y='y_dino_webm', size=10, source=stations_cds, color='orange', legend='Nearest location dinodata')
for ddl_id, station in stations_df.iterrows():
    for nap_feature in station['nap']:
        c = p.circle(
            x=nap_feature['properties']['x_webm'], 
            y=nap_feature['properties']['y_webm'],
            size=3,
            color='black',
            legend='nap point near tide gauge'
        )
# tools, so you can inspect and zoom in
p.add_tools(
    bokeh.models.HoverTool(
        renderers =[c1],
        tooltips=[
            ("name", "@Naam"),
            ("Lon, Lat", "(@lat, @lon)"),
        ]
    )
)

bokeh.plotting.show(p)

subsidence_df = pd.read_csv('../data/deltares/subsidence.csv').set_index('id')
nap_df = pd.read_csv('../data/deltares/subsidence_nap.csv').set_index('id')
subsidence_df.merge(nap_df, left_index=True, right_index=True)
stations_df.merge(subsidence_df, left_index=True, right_index=True)
subsidence_df

template = """
<%!
f3 = lambda x: "{:.3f}".format(float(x))
f0 = lambda x: "{:.0f}".format(float(x))
%>

<h2>${station['location']} <a id="${station.index}"></a></h2>

<style>
.right.template {
  float: right;
}
.template img {
  max-width: 300px !important;
}
</style>
<figure class="right template" >
    <img src="${station['img']}" />
    <figcaption>Photo of tide gauge at ${station['location']}, &copy; CIV, RWS</figcaption>
</figure>

<dl>
<dt>Location (lat, lon)</dt>
<dd>${station['lat'] | f3}, ${station['lon'] | f3}</dd>
<dt>Location (Rijksdriehoek)</dt>
<dd>${station['x_rd'] | f0}, ${station['y_rd'] | f0}</dd>
<dt>PSMSL-ID</dt>
<dd><a href="http://www.psmsl.org/data/obtaining/stations/${station['psmsl_id']}.php">${station['psmsl_id']}</a></dd>
<dt>Description</dt>
<dd>${station['summary']}</dd>
<dt>History</dt>
<dd><pre>${station['psmsl_info']}</pre></dd>
<dt>Nap info (public)</dt>
<dd><pre>
% for nap_feature in station['nap']:
${nap_feature['properties']['pub_tekst']} @ ${nap_feature['properties']['nap_hoogte']} (class: ${nap_feature['properties']['orde']}, distance: ${nap_feature['properties']['distance'] | f0}m)
% endfor
</pre></dd>
</dl>
<h2>Subsidence info</h2>
These are the estimated subsidence rates, based on the NAP history analysis in this repository and on the subsidence report of Hijma (2017). 
${subsidence.to_html()}
Other relevant links:
- <a href='http://a360.co/2s8ltK7'>Autocad drawing</a> of the construction.
"""

T = mako.template.Template(text=template)

def summary(station):
    return IPython.display.Markdown(
        T.render(
            station=stations_df.loc[station], 
            subsidence=pd.DataFrame(subsidence_df.loc[station])
        )
    )

def plot_station(code, stations_df=stations_df):
    station = stations_df.loc[code]
    
    # filter out masked data (no known litho)
    litho_ma = np.ma.masked_invalid(station['lithok']['litho'])
    litho = litho_ma[~litho_ma.mask]
    z = station['lithok']['z'][~litho_ma.mask]

    foundation_low = np.ma.masked_invalid(station['foundation_low'])
    station_low = np.ma.masked_invalid(station['station_low'])
    station_high = np.ma.masked_invalid(station['station_high'])

    fig = plt.figure(figsize=(13, 8))
    gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    ax1 = plt.subplot(gs[0])
    ax1.bar(
        np.zeros_like(z), 
        np.gradient(z), 
        0.8, 
        bottom=z,
        color=[colors[i] for i in litho]
    )    
    
    ax1.plot((0, 0), [foundation_low, station_low], 'k', lw=5, alpha=0.5)
    ax1.plot([0, 0], [station_low, station_high], 'k', lw=15, alpha=0.5)
    ax1.axhline(0, color='b', ls='--')
    ax1.set_title(('station ' + station.Naam))
    ax1.set_xlim(-0.1, 0.1)
    ax1.set_ylim(-50, 20)
    ax1.set_xticks([])
    ax1.set_ylabel('[m] relative to NAP')

    # plot in the 2nd axis to generate a legend
    ax2 = plt.subplot(gs[1])
    for label in labels:
        if label == 4:
            continue
        ax2.plot(0, 0, color=colors[label], label=labels[label])
    ax2.plot(0, 0, color='b', label='0m NAP', ls='--')
    ax2.legend(loc='center')
    ax2.axis('off')

station = 'DELFZL'
IPython.display.display(summary(station))
plot_station(station)

station = 'DENHDR'
IPython.display.display(summary(station))
plot_station(station)

station = 'HARLGN'
IPython.display.display(summary(station))
plot_station(station)

station = 'HOEKVHLD'
IPython.display.display(summary(station))
plot_station(station)

station = 'IJMDBTHVN'
IPython.display.display(summary(station))
plot_station(station)

station = 'VLISSGN' 
IPython.display.display(summary(station))
plot_station(station)

get_ipython().run_cell_magic('html', '', '<iframe src="https://api.mapbox.com/styles/v1/camvdvries/cj4o674wv8i9j2rs30ew26vm1.html?fresh=true&title=true&access_token=pk.eyJ1IjoiY2FtdmR2cmllcyIsImEiOiJjajA4NXdpNmswMDB2MzNzMjk4dGM2cnhzIn0.lIwd8N7wf0hx7mq-kjTcbQ#13.4/52.0524/4.1907/83.8/60" \nstyle="width: 100%; height: 500px;"/>')



