# important stuff:
import os
import pandas as pd
import numpy as np

import genpy
import gvars
import morgan as morgan

# stats
import sklearn.decomposition
import statsmodels.api as stm

# network graphics
import networkx as nx

# Graphics
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

# mcmc
import pymc3 as pm

# Magic function to make matplotlib inline;
get_ipython().magic('matplotlib inline')

# This enables SVG graphics inline. 
get_ipython().magic("config InlineBackend.figure_formats = {'png', 'retina'}")

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2, 
      'axes.labelsize': 18, 
      'axes.titlesize': 18, 
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style("dark")

mpl.rcParams['xtick.labelsize'] = 16 
mpl.rcParams['ytick.labelsize'] = 16 
mpl.rcParams['legend.fontsize'] = 14

q = 0.1
genvar = gvars.genvars()

# Specify the genotypes to refer to:
single_mutants = ['b', 'c', 'd', 'e', 'g']
# Specify which genotypes are double mutants 
double_mutants = {'a' : 'bd', 'f':'bc'}

# initialize the morgan.hunt object:
thomas = morgan.hunt('target_id', 'b', 'tpm', 'qval')
thomas.add_single_mutant(single_mutants)
thomas.add_double_mutants(['a', 'f'], ['bd', 'bc'])
thomas.add_genmap('../input/library_genotype_mapping.txt', comment='#')
thomas.set_qval()

# Add the tpm files: 
kallisto_loc = '../input/kallisto_all/'
sleuth_loc = '../sleuth/kallisto/'

thomas.add_tpm(kallisto_loc, '/kallisto/abundance.tsv', '')
# load all the beta dataframes:
for file in os.listdir("../sleuth/kallisto"):
    if file[:4] == 'beta':
        letter = file[-5:-4].lower()
        thomas.add_beta(sleuth_loc + file, letter)
        thomas.beta[letter].sort_values('target_id', inplace=True)
        thomas.beta[letter].reset_index(inplace=True)

thomas.filter_data()
# labelling var:
genes = [genvar.fancy_mapping[x] for x in thomas.single_mutants]

frames = []
for key, df in thomas.beta.items():
    df['genotype'] = genvar.fancy_mapping[key]
    df['code'] = key
    frames += [df]
tidy_data = pd.concat(frames)

# drop any genes that don't have a WormBase ID
tidy_data.dropna(subset=['ens_gene'], inplace=True)
# take a look at it:
tidy_data.head()

max_overlap = tidy_data[tidy_data.qval < q].target_id.unique()
print('There are {0} isoforms that are DE in at least one genotype in this analysis'.format(len(max_overlap)))

grouped = tidy_data.groupby('code')
bvals = np.array([])
labels = []
for code, group in grouped:
    # find names:
    names = group.target_id.isin(max_overlap)
    # extract (b, q) for each gene
    bs = group[names].b.values
    qs = group[names].qval.values
    
    # find sig genes:
    inds = np.where(qs > q)
    # set non-sig b values to 0
    bs[inds] = 0
    #standardize bs
    bs = (bs - bs.mean())/(bs.std())
    
    # place in array
    if len(bvals) == 0:
        bvals = bs
    else:
        bvals = np.vstack((bvals, bs))
    # make a label array
    labels +=  [code]

# initialize the PCA object and fit to the b-values
sklearn_pca = sklearn.decomposition.PCA(n_components=2).fit(bvals)
coords = sklearn_pca.fit(bvals).transform(bvals)

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', 'k']
shapes = ['D', 'D', 'v', '8', 'D', 'v', 'o']

# go through each pair of points and plot them:
for i, array in enumerate(coords):
    l = genvar.fancy_mapping[labels[i]]
    plt.plot(array[0], array[1], shapes[i], color=colors[i], label=l, ms=17)

# plot prettify:
plt.legend(loc=(1, 0.25), fontsize=16)
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.savefig('../output/PCA_genotypes.svg', bbox_inches='tight')

# the genotypes to compare
letters = ['e', 'b']
sig = tidy_data[(tidy_data.code.isin(letters)) & (tidy_data.qval < q)]
grouped = sig.groupby('target_id')
genes = []

# find the intersection between the two.
for target, group in grouped:
    # make sure the group contains all desired genotypes
    all_in = (len(group.code.unique()) == 2)
    if all_in:
        genes += [target]

# extract a temporary dataframe with all the desired genes
temp = tidy_data[tidy_data.target_id.isin(genes)]

# split the dataframes and find the rank of each gene
ovx = genpy.find_rank(temp[temp.code == letters[0]])
ovy = genpy.find_rank(temp[temp.code == letters[1]])

# Place data into dictionary:
data = dict(x=ovx.r, y=ovy.r)
x = np.linspace(ovx.r.min(), ovx.r.max())

# perform the simulation
trace_robust = genpy.robust_regress(data)

# find inliers, and outliers
outliers = genpy.find_inliers(ovx, ovy, trace_robust)

# run a second regression on the outliers
data2 = dict(x=ovx[ovx.target_id.isin(outliers)].r,
             y=ovy[ovy.target_id.isin(outliers)].r)

trace_robust2 = genpy.robust_regress(data2)
slope2 = trace_robust2.x.mean()

# draw a figure
plt.figure(figsize=(5, 5))

# plot mcmc results
label = 'posterior predictive regression lines'
pm.glm.plot_posterior_predictive(trace_robust, eval=x, 
                                 label=label, color='#357EC7')

# only plot secondary slope if it's of opposite sign to first
slope = trace_robust.x.mean()
if slope2*slope < 0:
    pm.glm.plot_posterior_predictive(trace_robust2, eval=x, 
                                     label=label, color='#FFA500')

# plot the data 
ind = ovx.target_id.isin(outliers)
x = ovx[~ind].r
y = ovy[~ind].r
plt.plot(x, y, 'go', ms = 5, alpha=0.4, label='inliers')

x = ovx[ind].r
y = ovy[ind].r
plt.plot(x, y, 'rs', ms = 6, label='outliers')

# prettify plot
plt.xlim(0, len(ovx))
plt.ylim(0, len(ovy))
plt.yticks([0, np.floor(len(ovx)/2), len(ovx)])
plt.xticks([0, np.floor(len(ovx)/2), len(ovx)])
plt.xlabel(genvar.fancy_mapping[letters[0]] +
           r'(lf) isoforms ranked by $\beta$')
plt.ylabel(genvar.fancy_mapping[letters[1]] +
           r'(lf) isoforms ranked by $\beta$')

comp = letters[0] + letters[1]
plt.savefig('../output/multiplemodes-{0}.svg'.format(comp), bbox_inches='tight')

# the genotypes to compare
letters = ['e', 'g']
sig = tidy_data[(tidy_data.code.isin(letters)) & (tidy_data.qval < q)]
grouped = sig.groupby('target_id')
genes = []

# find the intersection between the two.
for target, group in grouped:
    # make sure the group contains all desired genotypes
    all_in = (len(group.code.unique()) == 2)
    if all_in:
        genes += [target]

# extract a temporary dataframe with all the desired genes
temp = tidy_data[tidy_data.target_id.isin(genes)]

# split the dataframes and find the rank of each gene
ovx = genpy.find_rank(temp[temp.code == letters[0]])
ovy = genpy.find_rank(temp[temp.code == letters[1]])
plt.plot(ovx.r, ovy.r, 'go', ms = 5, alpha=0.4,)

plt.xlim(0, len(ovx))
plt.ylim(0, len(ovy))
plt.yticks([0, np.floor(len(ovx)/2), len(ovx)])
plt.xticks([0, np.floor(len(ovx)/2), len(ovx)])
plt.xlabel(genvar.fancy_mapping[letters[0]] +
           r'(lf) isoforms ranked by $\beta$')
plt.ylabel(genvar.fancy_mapping[letters[1]] +
           r'(lf) isoforms ranked by $\beta$')

barbara = morgan.mcclintock('bayesian', thomas, progress=False)

mat = barbara.robust_slope.as_matrix(columns=thomas.single_mutants)
labels = [genvar.fancy_mapping[x] for x in barbara.robust_slope.corr_with.values]

genpy.tri_plot(mat, labels)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.savefig('../output/bayes_primary_single_mutants.svg',
            bbox_inches='tight')

# make the graph:
G, width, weights, elarge = genpy.make_genetic_graph(barbara.robust_slope, w=3)

# paint the canvas:
with sns.axes_style('white'):
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G)  # positions for all nodes
    # draw the nodes:
    nx.draw_networkx_nodes(G, pos, node_size=1500,
                           node_color='g', alpha=.5)
    # draw the edges:
    edges = nx.draw_networkx_edges(G, pos, edgelist=elarge,
                                   width=width, edge_color=weights,
                                   edge_cmap=plt.cm.RdBu,
                                   edge_vmin=-.3, 
                                   edge_vmax=.3)
    # add the labels:
    nx.draw_networkx_labels(G, pos, font_size=16,
                            font_family='sans-serif')

    # add a colorbar:
    fig.colorbar(edges)
    sns.despine()
    sns.despine(left=True, bottom=True)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("../output/weighted_graph.svg") # save as png



