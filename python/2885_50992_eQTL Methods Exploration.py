import cPickle
import glob
import gzip
import os
import random
import shutil
import subprocess
import sys

import cdpybio as cpb
import matplotlib.pyplot as plt
import mygene
import myvariant
import numpy as np
import pandas as pd
import pybedtools as pbt
import scipy.stats as stats
import seaborn as sns

import ciepy
import cardipspy as cpy

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext rpy2.ipython')

dy_name = 'eqtl_methods_exploration'

import socket
if socket.gethostname() == 'fl-hn1' or socket.gethostname() == 'fl-hn2':
    dy = os.path.join(ciepy.root, 'sandbox', 'tmp', dy_name)
    cpy.makedir(dy)
    pbt.set_tempdir(dy)
    
outdir = os.path.join(ciepy.root, 'output', dy_name)
cpy.makedir(outdir)

private_outdir = os.path.join(ciepy.root, 'private_output', dy_name)
cpy.makedir(private_outdir)

fn = os.path.join(ciepy.root, 'output', 'input_data', 'rnaseq_metadata.tsv')
rna_meta = pd.read_table(fn, index_col=0)

fn = os.path.join(ciepy.root, 'output', 'eqtl_input', 'gene_to_regions.p')
gene_to_regions = cPickle.load(open(fn, 'rb'))
gene_info = pd.read_table(cpy.gencode_gene_info, index_col=0)

def orig_dir():
    os.chdir('/raid3/projects/CARDIPS/analysis/cardips-ipsc-eqtl/notebooks/')

fn = os.path.join(ciepy.root, 'output', 'eqtl_processing', 'eqtls01', 'qvalues.tsv')
qvalues = pd.read_table(fn, index_col=0)

fn = os.path.join(ciepy.root, 'output', 'input_data', 'rsem_tpm.tsv')
tpm = pd.read_table(fn, index_col=0)
fn = os.path.join(ciepy.root, 'output', 'input_data', 'rsem_expected_counts_norm.tsv')
ec = pd.read_table(fn, index_col=0)
fn = os.path.join(ciepy.root, 'output', 'input_data', 'gene_counts_norm.tsv')
gc = pd.read_table(fn, index_col=0)

tpm = tpm[rna_meta[rna_meta.in_eqtl].index]
tpm.columns = rna_meta[rna_meta.in_eqtl].wgs_id
tpm_sn = cpb.general.transform_standard_normal(tpm)
tpm_sn.to_csv(os.path.join(outdir, 'tpm_sn.tsv'), sep='\t')

ec = ec[rna_meta[rna_meta.in_eqtl].index]
ec.columns = rna_meta[rna_meta.in_eqtl].wgs_id
ec_sn = cpb.general.transform_standard_normal(ec)
ec_sn.to_csv(os.path.join(outdir, 'ec_sn.tsv'), sep='\t')

gc = gc[rna_meta[rna_meta.in_eqtl].index]
gc.columns = rna_meta[rna_meta.in_eqtl].wgs_id
gc_sn = cpb.general.transform_standard_normal(gc)
gc_sn.to_csv(os.path.join(outdir, 'gc_sn.tsv'), sep='\t')

fig,axs = plt.subplots(2, 2)
ax = axs[0, 0]
ax.scatter(tpm_sn.ix[qvalues.index[0]], ec_sn.ix[qvalues.index[0]])
ax.set_ylabel('Expected counts')
ax.set_xlabel('TPM')
ax = axs[0, 1]
ax.scatter(tpm_sn.ix[qvalues.index[0]], gc_sn.ix[qvalues.index[0]])
ax.set_ylabel('Gene counts')
ax.set_xlabel('TPM')
ax = axs[1, 0]
ax.scatter(gc_sn.ix[qvalues.index[0]], ec_sn.ix[qvalues.index[0]])
ax.set_ylabel('Expected counts')
ax.set_xlabel('Gene counts')
plt.tight_layout();

def make_emmax_sh(gene, exp, exp_name):
    out = os.path.join(private_outdir, '{}_{}'.format(gene, exp_name))
    cpy.makedir(out)
    f = open(os.path.join(out, '{}.sh'.format(gene)), 'w')
    f.write('#!/bin/bash\n\n')
    f.write('#$ -N emmax_{}_{}_test\n'.format(gene, exp_name))
    f.write('#$ -l opt\n')
    f.write('#$ -l h_vmem=2G\n')
    f.write('#$ -pe smp 4\n')
    f.write('#$ -S /bin/bash\n')
    f.write('#$ -o {}.out\n'.format(os.path.join(out, gene)))
    f.write('#$ -e {}.err\n\n'.format(os.path.join(out, gene)))

    f.write('module load cardips/1\n')
    f.write('source activate cie\n\n')

    f.write('python /frazer01/projects/CARDIPS/analysis/cardips-ipsc-eqtl/scripts/run_emmax.py \\\n')
    f.write('\t{} \\\n'.format(gene))
    f.write('\t/frazer01/projects/CARDIPS/analysis/cardips-ipsc-eqtl/private_output/eqtl_input/filtered_all/0000.vcf.gz \\\n')
    f.write('\t{} \\\n'.format(gene_to_regions[gene][0][3:]))
    f.write('\t{} \\\n'.format(exp))
    f.write('\t/frazer01/projects/CARDIPS/analysis/cardips-ipsc-eqtl/output/eqtl_input/emmax_samples.tsv \\\n')
    f.write('\t/frazer01/projects/CARDIPS/analysis/cardips-ipsc-eqtl/output/eqtl_input/wgs.kin \\\n')
    f.write('\t{} \\\n'.format(out))
    f.write('\t-c /frazer01/projects/CARDIPS/analysis/cardips-ipsc-eqtl/output/eqtl_input/emmax_full.tsv \\\n')
    f.write('\t-a 0\n')
    f.close()
    return os.path.join(out, '{}.sh'.format(gene))

genes = qvalues.index[0:5000:500]
names = ['tpm', 'ec', 'gc']
for gene in genes:
    for i,exp in enumerate([os.path.join(outdir, 'tpm_sn.tsv'), 
                            os.path.join(outdir, 'ec_sn.tsv'), 
                            os.path.join(outdir, 'gc_sn.tsv')]):
        exp_name = names[i]
        fn = make_emmax_sh(gene, exp, exp_name)
        get_ipython().system('qsub {fn}')

def plot_results(gene):
    gene = genes[0]
    fn = os.path.join(private_outdir, '{}_tpm'.format(gene), gene + '.tsv')
    tpm_res = ciepy.read_emmax_output(fn).dropna()
    fn = os.path.join(private_outdir, '{}_ec'.format(gene), gene + '.tsv')
    ec_res = ciepy.read_emmax_output(fn).dropna()
    fn = os.path.join(private_outdir, '{}_gc'.format(gene), gene + '.tsv')
    gc_res = ciepy.read_emmax_output(fn).dropna()
    for c in ['BETA', 'R2']:
        df = pd.DataFrame({'tpm': tpm_res[c], 'ec': ec_res[c], 'gc': gc_res[c]})
        sns.pairplot(df)
        plt.title(c)
    df = pd.DataFrame({'tpm': -np.log10(tpm_res['PVALUE']), 
                       'ec': -np.log10(ec_res['PVALUE']), 
                       'gc': -np.log10(gc_res['PVALUE'])})
    sns.pairplot(df)
    plt.title('$-\log_{10}$ $p$ value');

for g in genes:
    plot_results(g)

