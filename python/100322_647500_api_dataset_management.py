import requests
import json
import pandas as pd
from pprint import pprint
from multiprocessing import Pool

def f(dataset):
    data={}
    if dataset['attributes']['provider']!='wms':
        rasterUrl= 'https://api.resourcewatch.org/v1/query/'+dataset['id']+'?sql=select st_metadata(the_raster_webmercator) from '+dataset['attributes']['tableName']+' limit 1'
        geometryUrl='https://api.resourcewatch.org/v1/query/'+dataset['id']+'?sql=select * from '+dataset['attributes']['tableName']+' limit 1'
        url = geometryUrl if dataset['attributes']['provider']!='gee' or dataset['attributes']['tableName'][:3]=='ft:' else rasterUrl
        s = requests.get(url)
        if s.status_code!=200:
            data['dataset_id']=dataset['id']
            data['dataset_name']=dataset['attributes']['name']
            data['dataset_sql_status']=s.status_code
            data['connector_provider']=dataset['attributes']['provider']
            data['connector_url_status']=requests.get(dataset['attributes']['connectorUrl']).status_code if dataset['attributes']['provider']!='gee' else None
            data['connector_url']=dataset['attributes']['connectorUrl'] if dataset['attributes']['provider']!='gee' else dataset['attributes']['tableName']
            data['n_layers'] = len(dataset['attributes']['layer'])
            data['n_widgets'] = len(dataset['attributes']['widget'])
            return data
    else:
        for layer in dataset['attributes']['layer']:
            if 'url' in layer['attributes']['layerConfig']['body']:
                url = layer['attributes']['layerConfig']['body']['url']
                s = requests.get(url) 
                if s.status_code!=200:
                    data['dataset_id']=dataset['id']
                    data['dataset_name']=dataset['attributes']['name']
                    data['dataset_sql_status']=None
                    data['connector_provider']=dataset['attributes']['connectorUrl']
                    data['connector_url_status']=s.status_code
                    data['connector_url']=dataset['attributes']['connectorUrl']
                    data['n_layers'] = len(dataset['attributes']['layer'])
                    data['n_widgets'] = len(dataset['attributes']['widget'])
                    return data
        

def dataFrame(l,application):
    dDict={
    'dataset_id': [x['dataset_id'] for x in l if x!=None],
    'dataset_name': [x['dataset_name'] for x in l if x!=None],
    'dataset_sql_status': [x['dataset_sql_status'] for x in l if x!=None],
    'connector_provider': [x['connector_provider'] for x in l if x!=None],
    'connector_url_status': [x['connector_url_status'] for x in l if x!=None],
    'connector_url': [x['connector_url'] for x in l if x!=None],
    'n_layers': [x['n_layers'] for x in l if x!=None],
    'n_widgets': [x['n_widgets'] for x in l if x!=None]

    }
    pd.DataFrame(dDict).to_csv((application+'.csv'))
    return 'done'

def main(n, application):
    try:
        r = requests.get("https://api.resourcewatch.org/v1/dataset?application="+application+"&status=saved&includes=widget,layer&page%5Bsize%5D=14914800.35312")
    except requests.ConnectionError:
        print("Unexpected error:", requests.ConnectionError)
        raise
    else:
        dataset_list = r.json()['data']
        p = Pool(n)
        l = p.map(f, dataset_list)
        dataFrame(l,application)

main(20,'prep')

main(20,'rw')



