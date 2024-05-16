get_ipython().magic('matplotlib inline')

import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt

from qiime.parse import parse_mapping_file
from qiime.format import format_mapping_file
from skbio.io.util import open_file

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

get_ipython().run_cell_magic('bash', '', "\n# You must be in CU's VPN for the following to work\n\nmkdir -p gevers\n\nscp barnacle:/home/yovazquezbaeza/research/gevers/closed-ref-13-8/trimmed-100/otu_table.biom gevers/\nscp barnacle:/home/yovazquezbaeza/research/gevers/mapping_file.shareable.txt gevers/\n\nls gevers")

get_ipython().system("filter_samples_from_otu_table.py -i otu_table.biom -o otu_table.no-diarrhea.biom -s 'disease_stat:!acute hem. diarrhea,*' -m mapping-file-full.txt")

get_ipython().run_cell_magic('bash', '-e', '\nmkdir -p combined-gevers-suchodolski\n\n# both tables were picked against 13_8\nmerge_otu_tables.py \\\n-i otu_table.no-diarrhea.biom,gevers/otu_table.biom \\\n-o combined-gevers-suchodolski/otu-table.biom\n\nmerge_mapping_files.py \\\n-m mapping-file-full.txt,gevers/mapping_file.shareable.txt \\\n-o combined-gevers-suchodolski/mapping-file.txt \\\n--case_insensitive')

mf = load_mf('combined-gevers-suchodolski/mapping-file.txt')

def funk(row):
    if row['DIAGNOSIS'] == 'no_data':
        # we want to standardize the values of this column
        if row['DISEASE_STAT'] == 'healthy':
            return 'control'
        return row['DISEASE_STAT']
    else:
        return row['DIAGNOSIS']
mf['STATUS'] = mf.apply(funk, axis=1, reduce=True)

# clean up some other fields
repl = {'TITLE': {'no_data': 'Gevers_CCFA_RISK'},
        'HOST_COMMON_NAME': {'no_data': 'human'}}
mf.replace(repl, inplace=True)

write_mf('combined-gevers-suchodolski/mapping-file.standardized.txt',
         mf)

get_ipython().system('single_rarefaction.py -i combined-gevers-suchodolski/otu-table.biom -o combined-gevers-suchodolski/otu-table.15000.biom -d 15000')

