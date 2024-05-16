import networkx as nx
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

G = nx.read_gml('lesmiserables.gml', relabel=True)

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
nx.draw(G, node_size=0, edge_color='b', alpha=0.2, font_size=12, with_labels=True)

deg = nx.degree(G)
from numpy import percentile, mean, median
print min(deg.values())
print percentile(deg.values(),25) # computes the 1st quartile print median(deg.values())
print percentile(deg.values(),75) # computes the 3rd quartile print max(deg.values())

plt.hist(deg.values(), bins=40, range=(0,40))
plt.xlabel('Edges per Node')
plt.ylabel('Count')

sorted(deg.items(), key=lambda u: u[1], reverse=True)[:10]

G_sub = G.copy()
deg_sub = nx.degree(G_sub)
for n in G_sub.nodes():
    if deg_sub[n] < 10:
        G_sub.remove_node(n)

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
nx.draw(G_sub, node_size=0, edge_color='b', alpha=0.2, font_size=12, with_labels=True)

from networkx import find_cliques
cliques = list(find_cliques(G))

print max(cliques, key=lambda l: len(l))

