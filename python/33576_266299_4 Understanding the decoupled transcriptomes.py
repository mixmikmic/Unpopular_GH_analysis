# important stuff:
import os
import pandas as pd
import numpy as np

# morgan
import morgan as morgan
import tissue_enrichment_analysis as tea
import epistasis as epi
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
tissue_df = tea.fetch_dictionary()
phenotype_df = pd.read_csv('../input/phenotype_ontology.csv')
go_df = pd.read_csv('../input/go_dictionary.csv')

# Specify the genotypes to refer to:
single_mutants = ['b', 'c', 'd', 'e', 'g']
double_mutants = {'a' : 'bd', 'f':'bc'}

# initialize the morgan.hunt object:
thomas = morgan.hunt('target_id', 'b', 'tpm', 'qval')
thomas.add_genmap('../input/library_genotype_mapping.txt', comment='#')
thomas.add_single_mutant(single_mutants)
thomas.add_double_mutants(['a', 'f'], ['bd', 'bc'])
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
    df['genotype'] = genvar.fancy_mapping[key]
    df['code'] = key
    frames += [df]
tidy_data = pd.concat(frames)
tidy_data.dropna(subset=['ens_gene'], inplace=True)
tidy_data = tidy_data[tidy_data.code != 'g']

hif_genes = pd.read_csv('../output/hypoxia_response.csv')

n = len(hif_genes[hif_genes.b > 0].ens_gene.unique())
message = 'There are {0} unique genes that' +          ' are candidates for HIF-1 direct binding'
print(message.format(n))

ids = hif_genes[hif_genes.b > 0].target_id
hypoxia_direct_targets = tidy_data[tidy_data.target_id.isin(ids)]

names = hypoxia_direct_targets.sort_values('qval').target_id.unique()[0:10]

name_sort = {}
for i, name in enumerate(names):
    name_sort[name] = i+1

plot_df = tidy_data[tidy_data.target_id.isin(names)].copy()
plot_df['order'] = plot_df.target_id.map(name_sort)
plot_df.sort_values('order', inplace=True)
plot_df.reset_index(inplace=True)  

genpy.qPCR_plot(plot_df, genvar.plot_order, genvar.plot_color,
                clustering='genotype', plotting_group='target_id',
                rotation=90)

_ = tea.enrichment_analysis(hypoxia_direct_targets.ens_gene.unique(),
                            phenotype_df, show=True)

_ = tea.enrichment_analysis(hypoxia_direct_targets.ens_gene.unique(),
                            go_df, show=False)
tea.plot_enrichment_results(_, analysis='go')

# find the genes that overlap between vhl1 and egl-9vhl-1 and change in same directiom
vhl_pos = epi.find_overlap(['d', 'a'], positive)
vhl_neg = epi.find_overlap(['d', 'a'], negative)
vhl = list(set(vhl_pos + vhl_neg))

# find genes that change in the same direction in vhl(-) and vhl(+ datasets)
same_vhl = []
for genotype in ['b', 'e', 'f', 'c']:
    same_vhl += epi.find_overlap(['d', 'a', genotype], positive)
    same_vhl += epi.find_overlap(['d', 'a', genotype], negative)

# put it all together:
ind = (collate(vhl)) & (~collate(same_vhl))
vhl_regulated = tidy_data[ind & (tidy_data.code == 'd')]

n = len(vhl_regulated.ens_gene.unique())
message = 'There are {0} genes that appear to be ' +          'regulated in a hif-1-independent, vhl-1-dependent manner.'
print(message.format(n))

# begin plotting
names = vhl_regulated.sort_values('qval').target_id.unique()[0:10]
name_sort = {}
for i, name in enumerate(names):
    name_sort[name] = i+1

plot_df = tidy_data[tidy_data.target_id.isin(names)].copy()
plot_df['order'] = plot_df.target_id.map(name_sort)
plot_df.sort_values('order', inplace=True)
plot_df.reset_index(inplace=True)  

genpy.qPCR_plot(plot_df, genvar.plot_order, genvar.plot_color,
                clustering='genotype', plotting_group='target_id',
                rotation=90)

# save to file
cols = ['ext_gene', 'ens_gene', 'target_id', 'b', 'qval']
vhl_regulated[cols].to_csv('../output/vhl_1_regulated_genes.csv')

# genes that change in the same direction in egl, rhy and eglhif
egl_pos = epi.find_overlap(['e', 'b', 'f'], positive)
egl_neg = epi.find_overlap(['e', 'b', 'f'], negative)
egl = list(set(egl_pos + egl_neg))

cup = tidy_data[(tidy_data.code == 'c') & (tidy_data.b > 0)]
bdown = tidy_data[(tidy_data.code.isin(['e', 'b', 'f'])) & (tidy_data.b < 0)]
cdown = tidy_data[(tidy_data.code == 'c') & (tidy_data.b > 0)]
bup = tidy_data[tidy_data.code.isin(['e', 'b', 'f']) & (tidy_data.b < 0)]

antihif_1 = pd.concat([cup, bdown])
antihif_2 = pd.concat([cdown, bup])

antihif = []
for genotype in ['b', 'e', 'f']:
    temp = epi.find_overlap([genotype, 'c'], antihif_1)
    antihif += temp
    temp = epi.find_overlap([genotype, 'c'], antihif_2)
    antihif += temp

ind = collate(egl) & (collate(antihif)) & (~collate(same_vhl))

egl_regulated = tidy_data[ind & (tidy_data.code == 'b')]

n = egl_regulated.ens_gene.unique().shape[0]
print('There appear to be {0} egl-specific genes'.format(n))

egl_regulated[['ext_gene', 'b', 'qval']]

names = egl_regulated.sort_values('qval').target_id.unique()
name_sort = {}
for i, name in enumerate(names):
    name_sort[name] = i+1

plot_df = tidy_data[tidy_data.target_id.isin(egl_regulated.target_id.unique())].copy()
plot_df['order'] = plot_df.target_id.map(name_sort)
plot_df.sort_values('order', inplace=True)
plot_df.reset_index(inplace=True)  

plot_df = tidy_data[tidy_data.target_id.isin(egl_regulated.target_id.unique())].copy()
plot_df['order'] = plot_df.target_id.map(name_sort)
plot_df.sort_values('order', inplace=True)
plot_df.reset_index(inplace=True)  

genpy.qPCR_plot(plot_df, genvar.plot_order, genvar.plot_color,
                clustering='genotype', plotting_group='target_id',
                rotation=90)

plt.savefig('../output/egl9_downstream.pdf', bbox_inches='tight')



