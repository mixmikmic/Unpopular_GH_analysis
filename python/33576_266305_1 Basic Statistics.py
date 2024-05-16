# important stuff:
import os
import pandas as pd
import numpy as np

import morgan as morgan
import genpy
import gvars

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

mpl.rcParams['xtick.labelsize'] = 16 
mpl.rcParams['ytick.labelsize'] = 16 
mpl.rcParams['legend.fontsize'] = 14

q = 0.1
# this loads all the labels we need
genvar = gvars.genvars()

# Specify the genotypes to refer to:
single_mutants = ['b', 'c', 'd', 'e', 'g']

# Specify which letters are double mutants and their genotype
double_mutants = {'a' : 'bd', 'f':'bc'}

# initialize the morgan.hunt object:
thomas = morgan.hunt('target_id', 'b', 'tpm', 'qval')
# input the genmap file:
thomas.add_genmap('../input/library_genotype_mapping.txt', comment='#')
# add the names of the single mutants
thomas.add_single_mutant(single_mutants)
# add the names of the double mutants
thomas.add_double_mutants(['a', 'f'], ['bd', 'bc'])
# set the q-value threshold for significance to its default value, 0.1
thomas.set_qval()

# Add the tpm files: 
kallisto_loc = '../input/kallisto_all/'
sleuth_loc = '../sleuth/kallisto/'
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
    df['genotype'] = genvar.mapping[key]
    frames += [df]
    df['sorter'] = genvar.sort_muts[key]
tidy = pd.concat(frames)

# I will make a new column, called absb where I place the absolute val of b
tidy['absb'] = tidy.b.abs()

# sort_values according to their position in the sorter column
# (makes sure single mutants are clustered and doubles are clustered)
tidy.sort_values('sorter', inplace=True)
tidy.dropna(subset=['ens_gene'], inplace=True)

total_genes_id = tidy.target_id.unique().shape[0]
print("Total isoforms identified in all genotypes: {0}".format(total_genes_id))

print('Genotype: DEG')
for x in tidy.genotype.unique():
    # select the DE isoforms in the current genotype:
    sel = (tidy.qval < q) & (tidy.genotype == x)
    # extract the number of unique genes:
    s = tidy[sel].ens_gene.unique().shape[0]
    print(
"""{0}: {1}""".format(x, s))

sns.boxplot(x='genotype', y='absb', data=tidy[tidy.qval < q])
plt.yscale('log')
plt.xticks(rotation=30)

sig = (tidy.qval < q)
print('pair, shared GENES, percent shared (isoforms)')
for i, g1 in enumerate(tidy.genotype.unique()):
    genes1 = tidy[sig & (tidy.genotype == g1)]
    for j, g2 in enumerate(tidy.genotype.unique()[i+1:]):
        genes2 = tidy[sig & (tidy.genotype == g2)]
        
        # find the overlap between the two:
        n = genes2[genes2.ens_gene.isin(genes1.ens_gene)].shape[0]
        OR = ((tidy.genotype == g1) | (tidy.genotype == g2)) 
        
        n_iso = genes2[genes2.target_id.isin(genes1.target_id)].shape[0]
        ntot = tidy[sig & OR].target_id.shape[0]
        print(
            "{0}-{1}, {2}, {3:.2g}%".format(g1, g2, n, 100*n_iso/ntot)
             )

