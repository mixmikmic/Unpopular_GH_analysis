import pandas as pd
from pandas.io.json import json_normalize
import json, requests
import config

endpoint = 'https://ridb.recreation.gov/api/v1/organizations/{orgID}/recareas'
org_id = 128

offset=0
params = dict(apiKey= config.RIDB_API_KEY, offset=offset)
nps_url = endpoint.replace('{orgID}', str(org_id))
resp = requests.get(url=nps_url, params=params)
data = json.loads(resp.text)
df = json_normalize(data['RECDATA'])
df_nps = df

max_records = data['METADATA']['RESULTS']['TOTAL_COUNT']

df_nps.shape

while offset < max_records:
    offset = offset + len(df)
    print("offset: " + str(offset))
    df = pd.DataFrame()
    params = dict(apiKey= config.RIDB_API_KEY, offset=offset)
    try :
        resp = requests.get(url=nps_url, params=params)
    except Exception as ex:
        print(ex)
        break
    if resp.status_code == 200:
        data = json.loads(resp.text)
        if data['METADATA']['RESULTS']['CURRENT_COUNT'] > 0 :
            df = json_normalize(data['RECDATA'])
            df_nps = df_nps.append(df)
    else :
        print ("Response: " + str(resp.status_code))

df_nps.shape

df_np = df_nps[df_nps['RecAreaName'].apply(lambda x: x.find('National Park') > 0)]

df_np.shape

df_np[df_np['RecAreaLongitude'] == ""].RecAreaName

df_np[df_np['RecAreaName'] == 'Rocky Mountain National Park']

df_np[['RecAreaLatitude', 'RecAreaLongitude']].head()

missing_latlongs = pd.read_csv('missing_lat_longs.csv')
missing_latlongs.head()

missing_latlongs = missing_latlongs.set_index('RecAreaName')
df_np = df_np.set_index('RecAreaName')

df_np.update(missing_latlongs)

df_np[df_np['RecAreaLongitude'] == ""].index

df_np = df_np[df_np['RecAreaLongitude'] != ""]

df_np.shape

df_np['RecAreaName'] = df_np.index

df_np.shape

df_np['newIndex'] = range(0,58)

df_np.set_index(df_np['newIndex'])

df_np = df_np.drop('newIndex', axis=1)

df_np.shape

df_np.columns

df_np.to_csv('np_info.csv')

collection = {'type':'FeatureCollection', 'features':[]}

def feature_from_row(title, latitude, longitude):
    feature = { 'type': 'Feature', 
               'properties': { 'title': ''},
               'geometry': { 'type': 'Point', 'coordinates': []}
               }
    feature['geometry']['coordinates'] = [longitude, latitude]
    feature['properties']['title'] = title
    collection['features'].append(feature)
    return feature


geojson_series = df_np.apply(lambda x: feature_from_row(x['title'],x['latitude'],x['longitude'],x['description']),
                                  axis=1)

