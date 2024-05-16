import pandas as pd
import numpy as np
from pandas import json

cg_data = pd.read_csv('campgrounds.csv')

cg_data.shape

cg_data.head()

cg_data_clean = cg_data

cg_data_clean = cg_data_clean.replace({'flush': {'1':'Flush toilet', '0':'', '\\N':''}})
cg_data_clean = cg_data_clean.replace({'shower': {'1':'Shower', '0':'', '\\N':''}})
cg_data_clean = cg_data_clean.replace({'vault': {'1':'Vault toilet', '0':'', '\\N':''}})

cg_data_clean = cg_data_clean.rename(columns={'facilityname': 'title', 
                                              'facilitylatitude':'latitude', 
                                              'facilitylongitude':'longitude'})

cg_data_clean

cg_data_clean['description'] = cg_data_clean[['flush','shower','vault']].apply(lambda x: ', '.join(x), axis=1)

def clean_description(description):
    description = description.strip()
    while((description.startswith(',') or description.endswith(',')) and len(description) > -1):
        if description.endswith(',') :
            description = description[0:len(description)-1]
        if description.startswith(',') :
            description = description[1:len(description)]   
        description = description.strip()
    return description

cg_data_clean['description'] = cg_data_clean.description.apply(lambda x: clean_description(x))

cg_data_clean

geojson_df = cg_data_clean[['title','latitude','longitude','description']]

geojson_df

collection = {'type':'FeatureCollection', 'features':[]}

def feature_from_row(title, latitude, longitude, description):
    feature = { 'type': 'Feature', 
               'properties': { 'title': '', 'description': ''},
               'geometry': { 'type': 'Point', 'coordinates': []}
               }
    feature['geometry']['coordinates'] = [longitude, latitude]
    feature['properties']['title'] = title
    feature['properties']['description'] = description
    collection['features'].append(feature)
    return feature

geojson_series = geojson_df.apply(lambda x: feature_from_row(x['title'],x['latitude'],x['longitude'],x['description']),
                                  axis=1)

collection

with open('collection.geojson', 'w') as outfile:
    
    json.dump(collection, outfile)

test = pd.read_json('collection.geojson')

test



