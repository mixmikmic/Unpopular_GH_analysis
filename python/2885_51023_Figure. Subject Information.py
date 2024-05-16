import copy
import os
import subprocess

import cdpybio as cpb
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ciepy
import cardipspy as cpy

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext rpy2.ipython')

dy_name = 'figure_subject_information'
    
outdir = os.path.join(ciepy.root, 'output', dy_name)
cpy.makedir(outdir)

private_outdir = os.path.join(ciepy.root, 'private_output', dy_name)
cpy.makedir(private_outdir)

fn = os.path.join(ciepy.root, 'output', 'input_data', 'wgs_metadata.tsv')
wgs_meta = pd.read_table(fn, index_col=0, squeeze=True)
fn = os.path.join(ciepy.root, 'output', 'input_data', 'rnaseq_metadata.tsv')
rna_meta = pd.read_table(fn, index_col=0)
rna_meta = rna_meta[rna_meta.in_eqtl]
fn = os.path.join(ciepy.root, 'output', 'input_data', 'subject_metadata.tsv')
subject_meta = pd.read_table(fn, index_col=0)

subject_meta = subject_meta.ix[set(rna_meta.subject_id)]
family_vc = subject_meta.family_id.value_counts()
family_vc = family_vc[family_vc > 1]
eth_vc = subject_meta.ethnicity_group.value_counts().sort_values()
sex_vc = subject_meta.sex.value_counts()
sex_vc.index = pd.Series(['Female', 'Male'], index=['F', 'M'])[sex_vc.index]

sns.set_style('whitegrid')

p = subject_meta.ethnicity_group.value_counts()['European'] / float(subject_meta.shape[0])
print('{:.2f}% of the subjects are European.'.format(p * 100))

n = subject_meta.age.median()
print('Median subject age: {}.'.format(n))

p = sex_vc['Female'] / float(sex_vc.sum())
print('{:.2f}% of subjects are female.'.format(p * 100))

bcolor = (0.29803921568627451, 0.44705882352941179, 0.69019607843137254, 1.0)

fig = plt.figure(figsize=(6.85, 4), dpi=300)

gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
ax.text(0, 1, 'Figure S1',
        size=16, va='top')
ciepy.clean_axis(ax)
ax.set_xticks([])
ax.set_yticks([])
gs.tight_layout(fig, rect=[0, 0.90, 0.5, 1])

# Age
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
subject_meta.age.hist(bins=np.arange(5, 95, 5), ax=ax)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(8)
ax.set_xlabel('Age (years)', fontsize=8)
ax.set_ylabel('Number of subjects', fontsize=8)
ax.grid(axis='x')
gs.tight_layout(fig, rect=[0, 0.45, 0.4, 0.9])

# Ethnicity
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
eth_vc.plot(kind='barh', color=bcolor)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(8)
ax.set_ylabel('Ethnicity', fontsize=8)
ax.set_xlabel('Number of subjects', fontsize=8)
ax.grid(axis='y')
gs.tight_layout(fig, rect=[0.4, 0.45, 1, 0.9])

# Family size
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
family_vc.plot(kind='bar', color=bcolor)
ax.set_xticks([])
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(8)
ax.set_xlabel('Family', fontsize=8)
ax.set_ylabel('Number of family members', fontsize=8)
gs.tight_layout(fig, rect=[0.4, 0, 1, 0.45])

# Sex 
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
sex_vc.plot(kind='barh', color=bcolor)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(8)
ax.set_ylabel('Sex', fontsize=8)
ax.set_xlabel('Number of subjects', fontsize=8)
ax.grid(axis='y')
gs.tight_layout(fig, rect=[0, 0, 0.4, 0.45])

t = fig.text(0.005, 0.87, 'A', weight='bold', 
             size=12)
t = fig.text(0.4, 0.87, 'B', weight='bold', 
             size=12)
t = fig.text(0.005, 0.45, 'C', weight='bold', 
             size=12)
t = fig.text(0.4, 0.45, 'D', weight='bold', 
             size=12)

fig.savefig(os.path.join(outdir, 'subject_info.pdf'))
fig.savefig(os.path.join(outdir, 'subject_info.png'), dpi=300)

fs = 10
bcolor = (0.29803921568627451, 0.44705882352941179, 0.69019607843137254, 1.0)

fig = plt.figure(figsize=(6, 4), dpi=300)

# Age
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
subject_meta.age.hist(bins=np.arange(5, 95, 5), ax=ax)
for t in ax.get_xticklabels():
    t.set_fontsize(8)
for t in ax.get_yticklabels():
    t.set_fontsize(fs)
ax.set_xlabel('Age (years)', fontsize=fs)
ax.set_ylabel('Number of subjects', fontsize=fs)
ax.grid(axis='x')
gs.tight_layout(fig, rect=[0, 0.45, 0.4, 1])

# Ethnicity
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
eth_vc.plot(kind='barh', color=bcolor)
ax.set_xticks(ax.get_xticks()[0::2])
for t in ax.get_xticklabels():
    t.set_fontsize(8)
for t in ax.get_yticklabels():
    t.set_fontsize(fs)
ax.set_ylabel('Ethnicity', fontsize=fs)
ax.set_xlabel('Number of subjects', fontsize=fs)
ax.grid(axis='y')
gs.tight_layout(fig, rect=[0.4, 0.45, 1, 1])

# Family size
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
family_vc.plot(kind='bar', color=bcolor)
ax.set_xticks([])
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(fs)
ax.set_xlabel('Family', fontsize=fs)
ax.set_ylabel('Number of family members', fontsize=fs)
gs.tight_layout(fig, rect=[0.4, 0, 1, 0.5])

# Sex 
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0])
sex_vc.plot(kind='barh', color=bcolor)
for t in ax.get_xticklabels():
    t.set_fontsize(8)
for t in ax.get_yticklabels():
    t.set_fontsize(fs)
ax.set_ylabel('Sex', fontsize=fs)
ax.set_xlabel('Number of subjects', fontsize=fs)
ax.grid(axis='y')
gs.tight_layout(fig, rect=[0, 0, 0.4, 0.5])

fig.savefig(os.path.join(outdir, 'subject_info_presentation.pdf'))

