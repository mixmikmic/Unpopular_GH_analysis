#  Dependencies
import requests
from elasticsearch import Elasticsearch,helpers
import numpy as np
import uuid
import random
import json

es = Elasticsearch(['atlas-kibana.mwt2.org:9200'],
                                 timeout=10000)

es.ping()

print(es.info(), '\n', es.cluster.health())

# checking indices

indices = es.indices.get_aliases().keys()
print('Total No. of Indices: ',len(indices),'\n')
# print('\n', indices)
rucio=(index for index in indices if('rucio-events' in index))
network_weather = (index for index in indices if('network_weather-2017' in index))
rucio_indices = []
nws_indices = []
for event in network_weather:
    nws_indices.append(event)
for event in rucio:
    rucio_indices.append(event)
print('total NWS indices:',len(nws_indices),'\n')
print(nws_indices[0:5], '\n')

print('total rucio indices:',len(rucio_indices),'\n')
print(rucio_indices[0:5])

count=es.count(index='network_weather-2017*')
print('total documents : {}'.format( count['count']) )

nws_indices_dict = {}
for event in nws_indices:
    i = es.count(index=event)
    nws_indices_dict[event] = i['count']
# print('total data points:',sum(int(list(indices_dict.values()))))

print(nws_indices_dict)

def extract_data(index, query, scan_size, scan_step):
    resp = es.search(
    index = index,
    scroll = '20m',
    body = query,
    size = scan_step)

    sid = resp['_scroll_id']
    scroll_size = resp['hits']['total']
    print('total hits in {} : {}'.format(index,scroll_size))
    results=[]
    for hit in resp['hits']['hits']:
        results.append(hit)
    #plot_data_stats(results)
    steps = int((scan_size-scan_step)/ scan_step)

    # Start scrolling

    for i in range(steps):
        if i%10==0:
            print("Scrolling index : {} ; step : {} ...\n ".format(index,i))
        resp = es.scroll(scroll_id = sid, scroll = '20m')
        # Update the scroll ID
        sid = resp['_scroll_id']
        # Get the number of results that we returned in the last scroll
        scroll_size = len(resp['hits']['hits'])
        if i%10==0:
            print("scroll size: " + str(scroll_size))
        for hit in resp['hits']['hits']:
            results.append(hit)
    
    print("\n Done Scrolling through {} !! \n".format(index))
    results = pd.DataFrame(results)
    print(results.info(), '\n')
    return results

import numpy as np
import pandas as pd
nws= extract_data(index='network_weather-2017.8.10',query={}, scan_size=1000000, scan_step=10000)

nws.to_csv('nws.csv')

nws['_type'].unique()

nws['_type'].value_counts()

myquery = {
    "query": {
        "term": {
            '_type': 'throughput'
            }
        }
    }
es.search(index='network_weather-2017.8.10',body={}, size = 1)

query2={
    "query": {
        "term": {
            '_type': 'throughput'
            }
        }
    }
import pandas as pd
throughput= extract_data(index='network_weather-2017.8.10', query=query2, scan_size=15033, scan_step=10000)

throughput.head()

cond1 = throughput['destSite'] ==float('nan')
a= throughput[not(cond1)]
# cond2 = a['srcSite'] !=float('nan')
# a= a[cond2]

a.info()

a.head()
# type(a['destSite'][2])

