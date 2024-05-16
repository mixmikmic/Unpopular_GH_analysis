import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# airport coordinates taking from http://openflights.org/data.html
columns = ['AirportID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 'Latitude', 'Longitude', 'Altitude', 'Timezone', 'Type', 'Source']
coords = pd.read_csv('airports.csv', header=None, names=columns)
coords.head(3)

coords.info()

coords[coords.IATA == 'LAX']

coords[coords.City == 'Beverly']

fig, ax = plt.subplots(figsize=(10, 7))
plt.plot(coords.Longitude, coords.Latitude, 'k.')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

coords_us = coords[coords.Country == 'United States']

fig, ax = plt.subplots(figsize=(11, 9))
plt.plot(coords_us.Longitude, coords_us.Latitude, 'k.')
plt.plot([-70.916144], [42.584141], 'ro')
plt.xlim(-130, -65)
plt.ylim(20, 55)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('US Airport Locations')

# flight data is for April 2014
columns = ['flight_date', 'airline_id', 'flight_num', 'origin', 'destination', 'departure_time', 'departure_delay', 'arrival_time', 'arrival_delay', 'air_time', 'distance']
flights = pd.read_csv('../hadoop/ontime/flights.csv', parse_dates=[0], header=None, names=columns)
flights.head(3)

flights.info()

flights.flight_date.min(), flights.flight_date.max()

df = pd.merge(flights, coords_us, how='inner', left_on='origin', right_on='IATA')
df.shape

top50 = set(df.origin.value_counts()[:51].index.tolist())
top50.remove('HNL')
airports_top50 = coords_us[coords_us.IATA.isin(top50)]

fig, ax = plt.subplots(figsize=(11, 9))
plt.plot(airports_top50.Longitude, airports_top50.Latitude, 'k.')
plt.xlim(-130, -65)
plt.ylim(20, 55)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Top 50 US Airport Locations')

df50 = df[(df.origin.isin(top50)) & (df.destination.isin(top50))]

df50.head(3).transpose()

import networkx as nx

def make_node(x):
    return (x['IATA'], {'pos':(x['Longitude'], x['Latitude']), 'name':x['Name']})
airports_top50['nodes'] = airports_top50.apply(make_node, axis=1)

airports_top50.nodes[:5]

H = nx.DiGraph()
H.add_nodes_from(airports_top50.nodes.values)
pos = nx.get_node_attributes(H, 'pos')
fig, ax = plt.subplots(figsize=(11, 9))
nx.draw(H, pos, node_size=1000, edge_color='b', alpha=0.2, font_size=12, with_labels=True)

def f(x):
    return (x['origin'], x['destination'])
df50['edges'] = df50[['origin', 'destination']].apply(f, axis=1)

edge_counts = df50.edges.value_counts()
edge_counts[:5]

edges = []
for codes, count in zip(edge_counts.index, edge_counts.values):
    edges.append((codes[0], codes[1], count))

H.add_weighted_edges_from(edges)
fig, ax = plt.subplots(figsize=(11, 9))
nx.draw(H, pos, node_size=1000, edge_color='b', alpha=0.2, font_size=12, with_labels=True)

s_pagerank = pd.Series(nx.pagerank(H))
top10 = s_pagerank.sort_values(ascending=False)[:10]

for code, pr in top10.iteritems():
    print code, H.node[code]['name'], pr

s_flights = df50.origin.value_counts() + df50.destination.value_counts()
s_flights = s_flights / s_flights.sum()

ss = pd.DataFrame({'pagerank':s_pagerank, 'flights':s_flights})

fig, ax = plt.subplots(figsize=(10, 15))

ind = np.arange(ss.shape[0])
width = 0.4

ax.barh(ind, ss.flights.values, width, color='b', label='Total flights')
ax.barh(ind + width, ss.pagerank.values, width, color='r', label='Pagerank')
ax.set(yticks=ind + width, yticklabels=ss.index, ylim=[2 * width - 1, ss.shape[0]])
ax.legend()

nx.shortest_path(H, 'BOS', 'CMH')

pd.Series(nx.degree(H)).sort_values(ascending=False)[:10]

top50 - set(H.neighbors('ATL'))

H.number_of_edges(), H.number_of_nodes()

H.in_degree('ATL'), H.out_degree('ATL')

