get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext rmagic')

from __future__ import division

from qiime.parse import parse_mapping_file
from qiime.format import format_mapping_file
from skbio.io.util import open_file
from biom import load_table
from scipy.stats import mannwhitneyu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

def load_mf(fn):
    with open_file(fn, 'U') as f:
        mapping_data, header, _ = parse_mapping_file(f)
        _mapping_file = pd.DataFrame(mapping_data, columns=header)
        _mapping_file.set_index('SampleID', inplace=True)
    return _mapping_file

def write_mf(f, _df):
    with open_file(f, 'w') as fp:
        lines = format_mapping_file(['SampleID'] + _df.columns.tolist(),
                                    list(_df.itertuples()))
        fp.write(lines+'\n')

get_ipython().system('summarize_taxa.py -i otu_table.15000.25percent.biom -o stats/group-significance/taxa-summaries-25pct')

get_ipython().run_cell_magic('R', '', '\nlibrary("ccrepe")\n\notus <- read.table("stats/group-significance/taxa-summaries-25pct/otu_table.15000.25percent_L6.txt",\n                   sep="\\t", header=TRUE, skip=1, comment.char=\'\')\nrownames(otus) <- otus$X.OTU.ID\notus$X.OTU.ID <- NULL\n\notus.score <- ccrepe(x=t(otus), iterations=1000, sim.score=nc.score)\n\nwrite.table(otus.score$sim.score,\n            file=\'stats/group-significance/no-diarrhea/ccrepe/ccrepe-sim-score-otu_table.filtered.25pct_L6.txt\',\n            quote=FALSE, sep=\'\\t\')\n\nwrite.table(otus.score$z.stat,\n            file=\'stats/group-significance/no-diarrhea/ccrepe/ccrepe-z-stat-otu_table.filtered.25pct_L6.txt\',\n            quote=FALSE, sep=\'\\t\')\n\nwrite.table(otus.score$p.values,\n            file=\'stats/group-significance/no-diarrhea/ccrepe/ccrepe-p-values-otu_table.filtered.25pct_L6.txt\',\n            quote=FALSE, sep=\'\\t\')\n\nwrite.table(otus.score$q.values,\n            file=\'stats/group-significance/no-diarrhea/ccrepe/ccrepe-q-values-otu_table.filtered.25pct_L6.txt\',\n            quote=FALSE, sep=\'\\t\')')

otus_score = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-sim-score-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
z_stats = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-z-stat-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
p_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-p-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
q_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-q-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')

mask = np.zeros_like(otus_score, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# filter by the threshold
T = 0.4
mask = mask | ((otus_score < T) & (otus_score > (T*-1)))

# set a true value for all things that are not positive/negative
# in both matrices
es_filter_pos = (otus_score < 0) & (z_stats < 0)
es_filter_neg = (otus_score > 0) & (z_stats > 0)

# with the two matrices just remove whatever both matrices agreed
# on and save the data inside mask, we also want to use the pvalues
mask |= (es_filter_pos & es_filter_neg & (q_values > 0.05))

plt.figure(figsize=(50, 50))

g = sns.heatmap(otus_score, mask=mask, annot=True, fmt=".2f",
            annot_kws={'fontdict': {'fontsize': 8}})

plt.savefig('stats/group-significance/no-diarrhea/ccrepe/ccrepe-otu_table.filtered.25pct_L6.pdf')

get_ipython().system('group_significance.py -i stats/group-significance/taxa-summaries-25pct/otu_table.15000.25percent_L6.biom -o stats/group-significance/taxa-summaries-25pct/kruskall-wallis.txt -m mapping-file-full.txt --category disease_stat')

import networkx as nx

otus_score = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-sim-score-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
z_stats = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-z-stat-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
p_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-p-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
q_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-q-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')

mask = np.zeros_like(otus_score, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# filter by the threshold
T = 0.4
mask = mask | ((otus_score < T) & (otus_score > (T*-1)))

# set a true value for all things that are not positive/negative
# in both matrices
es_filter_pos = (otus_score < 0) & (z_stats < 0)
es_filter_neg = (otus_score > 0) & (z_stats > 0)

# with the two matrices just remove whatever both matrices agreed
# on and save the data inside mask, we also want to use the pvalues
mask |= (es_filter_pos & es_filter_neg & (q_values > 0.05))

kw_stats = pd.read_csv('stats/group-significance/taxa-summaries-25pct/kruskall-wallis.txt', sep='\t', index_col='OTU')
def color_funk(row):
    if row['healthy_mean'] > row['IBD_mean']:
        return 'g'
    else:
        return 'r'
colors = kw_stats.apply(color_funk, axis=1, reduce=False)

plt.figure(figsize=(20, 20))
G = nx.from_numpy_matrix(~mask.values)
G = nx.relabel_nodes(G, {i: o for i, o in enumerate(mask.index.tolist())})
G.remove_nodes_from([n for n in G.nodes_iter() if len(G.edges(n)) == 0])

for e in G.edges_iter():
    u, v = e
    weight = otus_score.loc[u][v]
    if weight > 0 :
        relation = 'coccurrence'
    else:
        relation = 'coexlcusion'

    G.add_edge(u, v, weight=weight, relation=relation)

nx.spring_layout(G)

#nx.draw(G, node_list=colors.index.tolist(), node_color=colors.tolist(), node_name=kw_stats.index.tolist())
nx.draw(G, node_list=colors.index.tolist(), node_color=colors.tolist())

node_attrs = pd.DataFrame()

kw_stats = pd.read_csv('stats/group-significance/taxa-summaries-25pct/kruskall-wallis.txt', sep='\t', index_col='OTU')
def color_funk(row):
    if row['healthy_mean'] > row['IBD_mean']:
        return 'protective'
    else:
        return 'inflammatory'
node_attrs['role'] = kw_stats.apply(color_funk, axis=1, reduce=False)

def short_name(row):
    #f__Planococcaceae;g__
    n = row.name.split('f__')[1]
    n = n.replace(';g__', ' ')
    
    if n.strip() == '':
        n = row.name.split('o__')[1].split(';')[0]
    return n
node_attrs['short_name'] = kw_stats.apply(short_name, axis=1, reduce=False)

node_attrs.to_csv('node-attributes.txt')

nx.write_edgelist(G, 'test.edgelist.2.txt', data=True)

import networkx as nx
import igraph

otus_score = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-sim-score-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
z_stats = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-z-stat-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
p_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-p-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
q_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-q-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')

mask = np.zeros_like(otus_score, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# filter by the threshold
T = 0.4
mask = mask | ((otus_score < T) & (otus_score > (T*-1)))

# set a true value for all things that are not positive/negative
# in both matrices
es_filter_pos = (otus_score < 0) & (z_stats < 0)
es_filter_neg = (otus_score > 0) & (z_stats > 0)

# with the two matrices just remove whatever both matrices agreed
# on and save the data inside mask, we also want to use the pvalues
mask |= (es_filter_pos & es_filter_neg & (q_values > 0.05))


kw_stats = pd.read_csv('stats/group-significance/taxa-summaries-25pct/kruskall-wallis.txt', sep='\t', index_col='OTU')
def color_funk(row):
    if row['healthy_mean'] > row['IBD_mean']:
        return 0x00ff00
    else:
        return 0xff0000
colors = kw_stats.apply(color_funk, axis=1, reduce=False)

# XXXXXX
connected_components = list(nx.connected_component_subgraphs(G))
largest_cc = max(connected_components, key=len)

nodes = {i: {'color': colors[i], 'size':1.25} for i in largest_cc.nodes_iter()}

edges = []
for edge in largest_cc.edges_iter():
    val = otus_score[edge[0]][edge[1]]
    if val > 0:
        edges.append({'source': edge[0], 'target': edge[1], 'color': 0x800080})
    elif val < 0:
        edges.append({'source': edge[0], 'target': edge[1], 'color': 0xFF8000})

graph = {
    'nodes': nodes,
    'edges': edges
}
igraph.draw(graph, directed=False)

import networkx as nx
import igraph

otus_score = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-sim-score-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
z_stats = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-z-stat-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
p_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-p-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')
q_values = pd.read_csv('stats/group-significance/no-diarrhea/ccrepe/ccrepe-q-values-otu_table.filtered.25pct_L6.txt',
                         sep='\t')

mask = np.zeros_like(otus_score, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# filter by the threshold
T = 0.4
mask = mask | ((otus_score < T) & (otus_score > (T*-1)))

# set a true value for all things that are not positive/negative
# in both matrices
es_filter_pos = (otus_score < 0) & (z_stats < 0)
es_filter_neg = (otus_score > 0) & (z_stats > 0)

# with the two matrices just remove whatever both matrices agreed
# on and save the data inside mask, we also want to use the pvalues
mask |= (es_filter_pos & es_filter_neg & (q_values > 0.05))

kw_stats = pd.read_csv('stats/group-significance/taxa-summaries-25pct/kruskall-wallis.txt',
                       sep='\t', index_col='OTU')

G = nx.from_numpy_matrix(~mask.values, parallel_edges=False, create_using=nx.Graph())
G = nx.relabel_nodes(G, {i: o for i, o in enumerate(mask.index.tolist())})
G.remove_nodes_from([n for n in G.nodes_iter() if len(G.edges(n)) == 0])

# find the largest connected component
connected_components = list(nx.connected_component_subgraphs(G))
largest_cc = max(connected_components, key=len)

good_bad = {'healthy': [], 'ibd': []}

for n in largest_cc.nodes():
    row = kw_stats.loc[n]
    
    if row['healthy_mean'] > row['IBD_mean']:
        good_bad['healthy'].append(n)
    else:
        good_bad['ibd'].append(n)

mf = load_mf('taxonomic_summaries/no-diarrhea/mapping-file-full.alpha_L6.txt')

mf['PD_whole_tree_even_15000_alpha'] = pd.to_numeric(mf.PD_whole_tree_even_15000_alpha, errors='coerce')

prot = set(good_bad['healthy'])
infl = set(good_bad['ibd'])

mf['Protective'] = pd.Series(np.zeros_like(mf.index.values), mf.index, dtype=np.float)
mf['Inflammatory'] = pd.Series(np.zeros_like(mf.index.values), mf.index, dtype=np.float)

for column_name in mf.columns:
    if any([True for p in prot if p in column_name]):
        mf['Protective'] += mf[column_name].astype(np.float)
    elif any([True for i in infl if i in column_name]):
        mf['Inflammatory'] += mf[column_name].astype(np.float)
    else:
        continue

# calculating the dysbiosis index
mf['Dogbyosis Index'] = np.divide(mf['Inflammatory'], mf['Protective']).astype(np.float)
# drop any samples with undefined values
mf['Dogbyosis Index'].replace({0: np.nan}, inplace=True)
mf['Dogbyosis Index'] = np.log(mf['Dogbyosis Index'])
mf.dropna(0, 'any', subset=['Dogbyosis Index'], inplace=True)

serializable_mf = mf.apply(lambda x: x.astype(str), axis=0)
write_mf('mapping-file.alpha.index.dogbyosis.txt', serializable_mf)

for k, v in good_bad.iteritems():
    print k
    print '\n'.join(sorted(good_bad[k]))

plt.figure()
sns.jointplot('Dogbyosis Index', 'PD_whole_tree_even_15000_alpha',
              mf[mf['disease_stat'] == 'healthy'], kind='reg', color='#1b9e77')
plt.savefig('md-index/new-md-index.healthy.pdf')

plt.figure()
sns.jointplot('Dogbyosis Index', 'PD_whole_tree_even_15000_alpha',
              mf[mf['disease_stat'] != 'healthy'], kind='reg', color='#d95f02')
plt.savefig('md-index/new-md-index.ibd.pdf')

get_ipython().system('make_emperor.py -i beta/15000/unweighted_unifrac_pc.txt -m  mapping-file.alpha.index.dogbyosis.txt -o beta/15000/unweighted-index --add_unique_columns')

get_ipython().system('make_emperor.py -i beta/15000/unweighted_unifrac_pc.txt -m mapping-file-full.alpha.L6index.txt -o beta/15000/unweighted-index-humans/ --add_unique_columns')



