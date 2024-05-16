get_ipython().magic('matplotlib inline')

import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt

from qiime.parse import parse_mapping_file
from qiime.format import format_mapping_file
from skbio.io.util import open_file
from scipy.stats import pearsonr, spearmanr
from skbio.stats.distance import permanova, anosim
from skbio import DistanceMatrix

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

mf = load_mf('mapping-file-full.alpha.txt')

mf.Antibiotics.value_counts()

get_ipython().system("filter_distance_matrix.py -i beta/15000/unweighted_unifrac_dm.txt -o beta/15000/unweighted_unifrac_dm.abxs-only.txt -m mapping-file-full.alpha.txt -s 'Antibiotics:definite_no,definite_yes'")

dm = DistanceMatrix.from_file('beta/15000/unweighted_unifrac_dm.abxs-only.txt')

emf = mf.loc[list(dm.ids)].copy()
emf.groupby('disease_stat').Antibiotics.value_counts()

permanova(dm.filter(emf[emf.disease_stat == 'IBD'].index, strict=False), mf, 'Antibiotics', permutations=10000)

permanova(dm.filter(emf[emf.disease_stat == 'healthy'].index, strict=False), mf, 'Antibiotics', permutations=10000)

permanova(DistanceMatrix.from_file('beta/15000/unweighted_unifrac_dm.txt'),
          mf, 'disease_stat', permutations=10000)

permanova(DistanceMatrix.from_file('beta/15000/unweighted_unifrac_dm.abxs-only.txt'),
          mf, 'Antibiotics', permutations=10000)



