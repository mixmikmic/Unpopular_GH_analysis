get_ipython().magic('matplotlib inline')

import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt

from qiime.parse import parse_mapping_file
from qiime.format import format_mapping_file
from skbio.io.util import open_file
from scipy.stats import pearsonr, spearmanr
from biom import load_table

from IPython.display import Image

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

def exploding_panda(_bt):
    """BIOM->Pandas dataframe converter

    Parameters
    ----------
    _bt : biom.Table
        BIOM table

    Returns
    -------
    pandas.DataFrame
        The BIOM table converted into a DataFrame
        object.
        
    References
    ----------
    Based on this answer on SO:
    http://stackoverflow.com/a/17819427/379593
    """
    m = _bt.matrix_data
    data = [pd.SparseSeries(m[i].toarray().ravel()) for i in np.arange(m.shape[0])]
    out = pd.SparseDataFrame(data, index=_bt.ids('observation'),
                             columns=_bt.ids('sample'))
    
    return out.to_dense()

get_ipython().run_cell_magic('bash', '', '\nmkdir -p stats/group-significance/no-diarrhea/\n\n# 5 percent\nfilter_otus_from_otu_table.py -s 8 \\\n-i otu_table.15000.no-diarrhea.biom \\\n-o stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.5pct.biom\n\n# 10 percent\nfilter_otus_from_otu_table.py -s 16 \\\n-i otu_table.15000.no-diarrhea.biom \\\n-o stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.10pct.biom\n\n# 20 percent\nfilter_otus_from_otu_table.py -s 32 \\\n-i otu_table.15000.no-diarrhea.biom \\\n-o stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.20pct.biom\n\n# 40 percent\nfilter_otus_from_otu_table.py -s 64 \\\n-i otu_table.15000.no-diarrhea.biom \\\n-o stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.40pct.biom')

get_ipython().run_cell_magic('bash', '-e', '\n# 5 percent\nsummarize_taxa.py \\\n-i stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.5pct.biom \\\n-o stats/group-significance/no-diarrhea/taxa-summaries-5pct/\n\n# genus\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-5pct/otu_table.15000.no-diarrhea.5pct_L6.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-5pct-L6.tsv \\\n-s kruskal_wallis\n\n# family\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-5pct/otu_table.15000.no-diarrhea.5pct_L5.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-5pct-L5.tsv \\\n-s kruskal_wallis\n\n# order\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-5pct/otu_table.15000.no-diarrhea.5pct_L4.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-5pct-L4.tsv \\\n-s kruskal_wallis\n\n# class\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-5pct/otu_table.15000.no-diarrhea.5pct_L3.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-5pct-L3.tsv \\\n-s kruskal_wallis\n\n# 10 percent\nsummarize_taxa.py \\\n-i stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.10pct.biom \\\n-o stats/group-significance/no-diarrhea/taxa-summaries-10pct/\n\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-10pct/otu_table.15000.no-diarrhea.10pct_L6.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-10pct-L6.tsv \\\n-s kruskal_wallis\n\n# 40 percent\nsummarize_taxa.py \\\n-i stats/group-significance/no-diarrhea/otu_table.15000.no-diarrhea.40pct.biom \\\n-o stats/group-significance/no-diarrhea/taxa-summaries-40pct/\n\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-40pct/otu_table.15000.no-diarrhea.40pct_L6.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-40pct-L6.tsv \\\n-s kruskal_wallis\n\ngroup_significance.py \\\n-i stats/group-significance/no-diarrhea/taxa-summaries-40pct/otu_table.15000.no-diarrhea.40pct_L4.biom \\\n-m mapping-file-full.alpha.L6index.txt \\\n-c disease_stat \\\n-o stats/group-significance/no-diarrhea/kruskall-wallis-40pct-L4.tsv \\\n-s kruskal_wallis')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sns.set(font="monospace")
sns.set_context("talk")

bt = load_table('stats/group-significance/no-diarrhea/taxa-summaries-40pct/otu_table.15000.no-diarrhea.40pct_L6.biom')
mf = load_mf('mapping-file-full.alpha.L6index.txt')
mf = mf.loc[bt.ids('sample')]

# dataframe from the group significance table
gsdf = pd.read_csv('stats/group-significance/no-diarrhea/kruskall-wallis-40pct-L6.tsv',
                   sep='\t')
gsdf = gsdf[gsdf['Bonferroni_P'] < 0.05]

current_palette = sns.color_palette()
cat_colors = dict(zip(mf.disease_stat.unique(),
                      current_palette[:3]))

# keep only the significant OTUs
bt.filter(gsdf.OTU.astype(str), axis='observation', inplace=True)
bt.norm()

df = exploding_panda(bt)

colors = []
for sid in bt.ids('sample'):
    colors.append(cat_colors[mf.loc[sid].disease_stat])

x = sns.clustermap(df, method="average",
                   figsize=(20, 20), col_colors=colors,
                   cmap=plt.get_cmap("Oranges"))

handles = []
for key, value in cat_colors.iteritems():
    handles.append(mpatches.Patch(color=value, label=key))

plt.legend(handles=handles)
# x.savefig('clustermap-L6.pdf')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sns.set(font="monospace")
sns.set_context("talk")

bt = load_table('stats/group-significance/no-diarrhea/taxa-summaries-10pct/otu_table.15000.no-diarrhea.10pct_L6.biom')

mf = load_mf('mapping-file-full.alpha.L6index.txt')
mf = mf.loc[bt.ids('sample')]

# dataframe from the group significance table
gsdf = pd.read_csv('stats/group-significance/no-diarrhea/kruskall-wallis-10pct-L6.tsv',
                   sep='\t')
gsdf = gsdf[gsdf['Bonferroni_P'] < 0.05]

current_palette = sns.color_palette()
cat_colors = dict(zip(mf.disease_stat.unique(),
                      current_palette[:3]))

# keep only the significant OTUs
bt.filter(gsdf.OTU.astype(str), axis='observation', inplace=True)
bt.norm()

df = exploding_panda(bt)

sample_order = mf[mf.disease_stat == 'healthy'].index.tolist() + mf[mf.disease_stat != 'healthy'].index.tolist()
df = df[sample_order]

colors = []
for sid in df.columns:
    colors.append(cat_colors[mf.loc[sid].disease_stat])

x = sns.clustermap(df, method="average",
                   row_cluster=True, col_cluster=False,
                   figsize=(20, 20), col_colors=colors,
                   cmap=plt.get_cmap("Oranges"))

handles = []
for key, value in cat_colors.iteritems():
    handles.append(mpatches.Patch(color=value, label=key))

plt.legend(handles=handles)
# x.savefig('clustermap-L6.pdf')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sns.set(font="monospace")
sns.set_context("talk")

bt = load_table('stats/group-significance/no-diarrhea/taxa-summaries-5pct/otu_table.15000.no-diarrhea.5pct_L4.biom')

mf = load_mf('mapping-file-full.alpha.L6index.txt')
mf = mf.loc[bt.ids('sample')]

# dataframe from the group significance table
gsdf = pd.read_csv('stats/group-significance/no-diarrhea/kruskall-wallis-5pct-L4.tsv',
                   sep='\t')
gsdf = gsdf[gsdf['Bonferroni_P'] < 0.05]

current_palette = sns.color_palette()
cat_colors = dict(zip(mf.disease_stat.unique(),
                      current_palette[:3]))

# keep only the significant OTUs
bt.filter(gsdf.OTU.astype(str), axis='observation', inplace=True)
bt.norm()

df = exploding_panda(bt)

sample_order = mf[mf.disease_stat == 'healthy'].index.tolist() + mf[mf.disease_stat != 'healthy'].index.tolist()
df = df[sample_order]

colors = []
for sid in df.columns:
    colors.append(cat_colors[mf.loc[sid].disease_stat])

x = sns.clustermap(df, method="average",
                   row_cluster=True, col_cluster=False,
                   figsize=(20, 20), col_colors=colors,
                   cmap=plt.get_cmap("Oranges"))

handles = []
for key, value in cat_colors.iteritems():
    handles.append(mpatches.Patch(color=value, label=key))

plt.legend(handles=handles)
# x.savefig('clustermap-L6.pdf')

