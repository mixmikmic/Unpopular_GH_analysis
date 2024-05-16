get_ipython().magic('matplotlib inline')
from __future__ import division

import pandas as pd
import numpy as np

from biom import load_table

import matplotlib.pyplot as plt

get_ipython().system('scp barnacle:/home/yovazquezbaeza/research/suchodolski-dogs/open-ref-otus/otu_table_mc2_w_tax.biom  counts/open-ref-table.biom')

get_ipython().system('biom summarize-table -i counts/open-ref-table.biom -o counts/open-ref-table-summary.txt')

sl = pd.read_csv('counts/split_library_log.txt', sep='\t', index_col='#SampleID')
ot = pd.read_csv('counts/table-summary.txt', sep='\t', index_col='#SampleID')
ot_open = pd.read_csv('counts/open-ref-table-summary.txt', sep='\t', index_col='#SampleID')

tot = pd.DataFrame(index=sl.index, columns=['Sequences', 'OTU_Counts', 'OTU_Counts_open'])

tot['Sequences'] = sl.Count
tot['OTU_Counts'] = ot.Count
tot['OTU_Counts_open'] = ot_open.Count

tot['Percent'] = np.divide(tot.OTU_Counts, tot.Sequences *1.0) * 100
tot['Percent_open'] = np.divide(tot.OTU_Counts_open, tot.Sequences *1.0) * 100

tot['Percent'].hist(bins=200)

plt.xlabel('Percent of Sequences Assigned to an OTU (closed reference)')
plt.ylabel('Counts')
plt.xlim([0, 100])

plt.title('Entire Cohort')

tot['Percent_open'].hist(bins=200)

plt.xlabel('Percent of Sequences Assigned to an OTU (open reference)')
plt.ylabel('Counts')
plt.xlim([0, 100])

plt.title('Entire Cohort')

bt = load_table('otu_table.15000.no-diarrhea.biom')

sub = tot.loc[bt.ids('sample')].copy()

sub['Percent'] = np.divide(sub.OTU_Counts, sub.Sequences *1.0) * 100
sub['Percent_open'] = np.divide(sub.OTU_Counts_open, sub.Sequences *1.0) * 100

sub['Percent'].hist(bins=200)

plt.xlabel('Percent of Sequences Assigned to an OTU')
plt.ylabel('Counts')
plt.xlim([70, 100])
plt.title('Samples used for Analysis')

sub['Percent_open'].hist(bins=200)

plt.xlabel('Percent of Sequences Assigned to an OTU (Open Reference)')
plt.ylabel('Counts')
plt.xlim([70, 100])
plt.title('Samples used for Analysis')

sub.Percent.mean()

sub.Percent_open.mean()



