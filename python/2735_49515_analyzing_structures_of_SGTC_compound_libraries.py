# numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import matplotlib.cm as cm

# scipy
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as hc

# seaborn -- for better looking plots
import seaborn as sns

# pandas 
import pandas as pd

# rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdmolops
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors

url = 'http://sgtc-svr102.stanford.edu/hiplab/compounds/browse/2/'

list_of_tables = pd.read_html(url)

# the _only_ table on the page is the first table in the list
table = list_of_tables[0]

smiles_list = table['SMILES']

mols = []
fingerprints = []

for idx, smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mols.append(mol)
    else:
        print 'Unable to parse item %s, SMILES string %s and so this molecule will be skipped.' % (idx, smiles)
        print 'Please check validity of this string or manually remove it from smiles_list.'
        continue
    
fingerprint_mat = np.vstack(np.asarray(rdmolops.RDKFingerprint(mol, fpSize = 2048), dtype = 'bool') for mol in mols)

smiles_list.pop(215)

dist_mat = pdist(fingerprint_mat, 'jaccard')

dist_df = pd.DataFrame(squareform(dist_mat), index = smiles_list, columns= smiles_list)

# set a mask
mask = np.zeros_like(dist_df, dtype = 'bool')
mask[np.triu_indices_from(mask, k = 1)] = True

# plot
ax = sns.heatmap(dist_df, mask=mask, cmap = cm.viridis, xticklabels=20, yticklabels=20, )
ax.tick_params(axis='both', which='major', labelsize=3)

# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
z = hc.linkage(dist_mat, metric='jaccard')
plt.figure(figsize=[4, 20])
dendrogram = hc.dendrogram(z, 
                           orientation = 'left',

                           labels = dist_df.columns,
                           show_leaf_counts = True,
                           show_contracted = True,
                           leaf_font_size = 2
                          )

plt.show()

# reorder dist_df according to clustering results
new_order = dendrogram['ivl']

# reorder both rows and columns
reordered_dist_df = dist_df[new_order].reindex(new_order)

# plot again
ax = sns.heatmap(reordered_dist_df, mask=mask, cmap = cm.viridis, xticklabels=20, yticklabels=20)
ax.tick_params(axis='both', which='major', labelsize=3)

low_distance = np.where(np.logical_and(dist_df <= 0.02,
                                       dist_df > 0)
                        )

similar_smiles_pairs = [indices for indices in zip(low_distance[0], low_distance[1]) 
                                          if indices[0] < indices[1]]

f, ax = plt.subplots(len(similar_smiles_pairs), 2, figsize = (10, 90))

for idx, pair in enumerate(similar_smiles_pairs):
    mol1 = Chem.MolFromSmiles(smiles_list[pair[0]])
    mol2 = Chem.MolFromSmiles(smiles_list[pair[1]])
    ax[idx, 0].imshow(Draw.MolToImage(mol1, size=(200, 200), fitImage=True))
    ax[idx, 1].imshow(Draw.MolToImage(mol2, size=(200, 200), fitImage=True))
    ax[idx, 0].grid(False)
    ax[idx, 1].grid(False)
    ax[idx, 0].axis('off')
    ax[idx, 1].axis('off')



