# important stuff:
import os
import pandas as pd
import numpy as np

import tissue_enrichment_analysis as tea
import morgan as morgan
import epistasis as epi
import genpy

# Graphics
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{cmbright}')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

# Magic function to make matplotlib inline;
get_ipython().magic('matplotlib inline')

# This enables SVG graphics inline. 
# There is a bug, so uncomment if it works.
get_ipython().magic("config InlineBackend.figure_formats = {'png', 'retina'}")

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2, 
      'axes.labelsize': 18, 
      'axes.titlesize': 18, 
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style("dark")

ft = 35 #title fontsize
import genpy
import gvars

mpl.rcParams['xtick.labelsize'] = 16 
mpl.rcParams['ytick.labelsize'] = 16 
mpl.rcParams['legend.fontsize'] = 14

tfs = pd.read_csv('../input/tf_list.csv')

q = 0.1
# this loads all the labels we need
genvar = gvars.genvars()

# Add the tpm files: 
kallisto_loc = '../input/kallisto_all/'
sleuth_loc = '../sleuth/kallisto/'
# Specify the genotypes to refer to:
single_mutants = ['b', 'c', 'd', 'e', 'g']
double_mutants = {'a' : 'bd', 'f':'bc'}

# initialize the morgan.hunt object:
thomas = morgan.hunt('target_id', 'b', 'tpm', 'qval')
thomas.add_genmap('../input/library_genotype_mapping.txt', comment='#')
thomas.add_single_mutant(single_mutants)
thomas.add_double_mutants(['a', 'f'], ['bd', 'bc'])
thomas.set_qval()
thomas.add_tpm(kallisto_loc, '/kallisto/abundance.tsv', '')

# load all the beta values for each genotype:
for file in os.listdir("../sleuth/kallisto"):
    if file[:4] == 'beta':
        letter = file[-5:-4].lower()
        thomas.add_beta(sleuth_loc + file, letter)
        thomas.beta[letter].sort_values('target_id', inplace=True)
        thomas.beta[letter].reset_index(inplace=True)
thomas.filter_data()

frames = []
for key, df in thomas.beta.items():
    df['genotype'] = genvar.fancy_mapping[key]
    df['code'] = key
    frames += [df]
    df['sorter'] = genvar.sort_muts[key]

tidy = pd.concat(frames)
tidy.sort_values('sorter', inplace=True)
tidy.dropna(subset=['ens_gene'], inplace=True)

codes = ['a', 'b', 'c', 'd', 'e', 'f']

print('Genotype, #TFs')
for c in codes:
    ind = (tidy.qval < q) & (tidy.code == c) & (tidy.target_id.isin(tfs.target_id))
    print(genvar.mapping[c], tidy[ind].shape[0])

# extract the hypoxia response:
hyp_response_pos = epi.find_overlap(['e', 'b', 'a', 'd'], tidy[tidy.b > 0])
hyp_response_neg = epi.find_overlap(['e', 'b', 'a', 'd'], tidy[tidy.b < 0])
hyp_response = list(set(hyp_response_neg + hyp_response_pos))

# find tfs in the hif-1 response
tfs_in_hif = tfs[tfs.target_id.isin(hyp_response)].target_id
print('There are {0} transcription factors in HIF-1+ animals'.format(tfs_in_hif.shape[0]))

# The qPCR function I wrote is quite stupid, so I always have to tidy up my dataframe a little
# bit and add a couple of columns:

# select the data to be plotted:
plotdf = tidy[tidy.target_id.isin(tfs_in_hif)].copy()
# sort by genotype
plotdf.sort_values(['genotype', 'target_id'], inplace=True)
# add an 'order' column
plot_order = {i: t+1 for t, i in enumerate(plotdf.target_id.unique())}
plotdf['order'] = plotdf.target_id.map(plot_order)
# sort by 'order'
plotdf.sort_values('order', inplace=True)
plotdf.reset_index(inplace=True)  

genpy.qPCR_plot(plotdf[plotdf.code != 'g'], genvar.plot_order, genvar.plot_color, clustering='genotype',
                plotting_group='target_id', rotation=90)

# extract the hypoxia response:
hyp_response_pos = epi.find_overlap(['e', 'b', 'a'], tidy[tidy.b > 0])
hyp_response_neg = epi.find_overlap(['e', 'b', 'a'], tidy[tidy.b < 0])
hyp_response = list(set(hyp_response_neg + hyp_response_pos))

print('There are {0} isoforms in the relaxed hypoxia response'.format(len(hyp_response)))
tfs_in_hif = tfs[tfs.target_id.isin(hyp_response)].target_id
print('There are {0} transcription factors in HIF-1+/HIF-1OH- animals'.format(tfs_in_hif.shape[0]))

plotdf = tidy[tidy.target_id.isin(tfs_in_hif)].copy()
plotdf.sort_values(['genotype', 'target_id'], inplace=True)
plot_order = {i: t+1 for t, i in enumerate(plotdf.target_id.unique())}
plotdf['order'] = plotdf.target_id.map(plot_order)
plotdf.sort_values('order', inplace=True)
plotdf.reset_index(inplace=True)  
plotdf = plotdf[['target_id', 'ens_gene', 'ext_gene','b', 'se_b', 'qval', 'genotype', 'order', 'code']]

genpy.qPCR_plot(plotdf[plotdf.code != 'g'], genvar.plot_order, genvar.plot_color, clustering='genotype',
                plotting_group='target_id', rotation=90)



