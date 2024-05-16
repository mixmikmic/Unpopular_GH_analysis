get_ipython().magic('matplotlib inline')
import requests
import json
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import zipfile
import os

def zipDir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            print(os.path.join(root, file))
            ziph.write(os.path.join(root, file))
            os.remove(os.path.join(root, file))

url='https://api.openaq.org/v1/latest'
payload = {
    'limit':10000,
    'has_geo':True
}
r = requests.get(url, params=payload)
r.status_code

from pandas.io.json import json_normalize
data = r.json()['results']
df = json_normalize(data, ['measurements'],[['coordinates', 'latitude'], ['coordinates', 'longitude'],'location', 'city', 'country'])

print(df.columns.values)
df.head(2)

geometry = [Point(xy) for xy in zip(df['coordinates.longitude'], df['coordinates.latitude'])]
df = df.drop(['coordinates.longitude', 'coordinates.latitude'], axis=1)
crs = {'init': 'epsg:4326'}
geo_df = GeoDataFrame(df, crs=crs, geometry=geometry)

geo_df.head(2)

#geo_df.plot();

geo_df.parameter.unique()

def export2shp(data, outdir, outname):
    current = os.getcwd()
    path= current+outdir
    os.mkdir(path)
    data.to_file(filename=(path+'/'+outname+'.shp'),driver='ESRI Shapefile')
    with zipfile.ZipFile(outname+'.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipDir(path, zipf)
    os.rmdir(path)

outdir='/dst'
outname='test'
export2shp(geo_df, outdir, outname)

import folium
from vega import Vega

urlw='https://api.resourcewatch.org/widget/ea0ecd72-41f4-4ced-965c-c95204174048'
wr = requests.get(urlw)
wr.json()['data']['attributes']['widgetConfig']

conf = requests.get('https://raw.githubusercontent.com/resource-watch/resource-watch-app/master/src/utils/widgets/vega-theme-thumbnails.json').json()
spec = wr.json()['data']['attributes']['widgetConfig']

    
t = Vega(spec)
t.config = conf

t = []
path=''
def find_value(dic, val, path):
        for key, value in dic.items():
            if value == val:
                path = path +'[\''+ key + '\']'
                t.append(path)
            elif isinstance(dic[key], list):
                for i, data in enumerate(dic[key]):
                    if isinstance(data, dict):
                        symnpath = path +'[\''+ key +'\']['+ str(i)+']'
                        find_value(data, val, symnpath)
            elif isinstance(dic[key], dict):
                symnpath = path +'[\''+ key+'\']'
                find_value(dic[key], val, symnpath)

find_value(spec, 'colorRange1', path)
print(t[0])
fs = t[0]

d = 'spec' + t[0]
print(d)
sdf = eval(d)
sdf

exec("d=conf['range'][sdf]")

spec['scales'][2]['range']='category10'

Vega(spec)

account = 'wri-rw'
urlCarto = 'https://'+account+'.carto.com/api/v1/map'
body = {
    "layers": [{
        "type": "cartodb",
        "options": {
            "sql": "select * from countries",
            "cartocss":"#layer {\n  polygon-fill: #374C70;\n  polygon-opacity: 0.9;\n  polygon-gamma: 0.5;\n  line-color: #FFF;\n  line-width: 1;\n  line-opacity: 0.5;\n  line-comp-op: soft-light;\n}",
            "cartocss_version": "2.1.1"
        }
    }]
}

r = requests.post(urlCarto, data=json.dumps(body), headers={'content-type': 'application/json; charset=UTF-8'})
tileUrl = 'https://'+account+'.carto.com/api/v1/map/' + r.json()['layergroupid'] + '/{z}/{x}/{y}.png32';

map_osm = folium.Map(location=[45.5236, 0.6750], zoom_start=3)
folium.TileLayer(
    tiles=tileUrl,
    attr='text',
    name='text',
    overlay=True
).add_to(map_osm)
map_osm



