get_ipython().system('head ./wiki/wiki-links-abridged.txt')

get_ipython().system('head ./wiki/wiki-links-simple-sorted.txt')

get_ipython().system('head ./wiki/wiki-titles-sorted.txt')

import networkx as nx

graph_dict = dict()
l = dict()

with open("./wiki/wiki-links-simple-sorted.txt", "r") as g, open("./wiki/wiki-titles-sorted.txt", "r") as f:
    for line in g:
        text = f.readline()[:-1]
#         print(line, text)
        vertex = int(line.split(": ")[0])
        edges = line.split(":")[1].split()
        edges = set([int(edge) for edge in edges])
#         print(graph_dict, edges)
        graph_dict[vertex] = edges
        l[vertex] = text
        if vertex%300000 == 0:
            print(vertex)
#         if vertex == 10:
#             break
#     print(g)
#     print(l)
    print("Done")
    G=nx.Graph(graph_dict)
    result = nx.closeness_centrality(G)

with open("../Texts/french_expressions.txt", "r") as f:
    for line in f:
        if len(line) > 1 and line[-2] in r'.]:)"' or ";" in line:
            continue
        print(line)

import pandas as pd

df = pd.read_csv("../Texts/latin_expressions.txt", sep='\t')

df.head()



