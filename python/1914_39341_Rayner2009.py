import warnings
warnings.filterwarnings("ignore")

from astropy.io import ascii

import pandas as pd

#! curl http://iopscience.iop.org/0067-0049/185/2/289/suppdata/apjs311476t7_ascii.txt > ../data/Rayner2009/apjs311476t7_ascii.txt

#! head ../data/Rayner2009/apjs311476t7_ascii.txt

nn = ['wl1', 'id1', 'wl2', 'id2', 'wl3', 'id3', 'wl4', 'id4']

tbl7 = pd.read_csv("../data/Rayner2009/apjs311476t7_ascii.txt", index_col=False,
                   sep="\t", skiprows=[0,1,2,3], names= nn)

line_list_unsorted = pd.concat([tbl7[[nn[0], nn[1]]].rename(columns={"wl1":"wl", "id1":"id"}),
           tbl7[[nn[2], nn[3]]].rename(columns={"wl2":"wl", "id2":"id"}),
           tbl7[[nn[4], nn[5]]].rename(columns={"wl3":"wl", "id3":"id"}),
           tbl7[[nn[6], nn[7]]].rename(columns={"wl4":"wl", "id4":"id"})], ignore_index=True, axis=0)

line_list = line_list_unsorted.sort_values('wl').dropna().reset_index(drop=True)

#line_list.tail()

sns.distplot(line_list.wl)

line_list.to_csv('../data/Rayner2009/tbl7_clean.csv', index=False)

