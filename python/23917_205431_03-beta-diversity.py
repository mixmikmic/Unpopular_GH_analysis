get_ipython().magic('matplotlib inline')

import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt

from qiime.parse import parse_mapping_file
from qiime.format import format_mapping_file
from skbio.io.util import open_file
from scipy.stats import pearsonr, spearmanr

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

get_ipython().run_cell_magic('bash', '', '\ncurl -O ftp://greengenes.microbio.me/greengenes_release/gg_13_5/gg_13_8_otus.tar.gz\ntar -xzf gg_13_8_otus.tar.gz')

get_ipython().system('beta_diversity_through_plots.py -i otu_table.15000.no-diarrhea.biom -m mapping-file-full.alpha.L6index.txt -t gg_13_8_otus/trees/97_otus.tree -o beta/15000 -a -O 7 --color_by_all_fields -f')

Image('beta/15000/screen-shots/faecalibacterium-unweighted-diseased-are-big-spheres.png')

Image('beta/15000/screen-shots/unweighted-disease-status.png')

get_ipython().system('split_otu_table.py -i otu_table.15000.no-diarrhea.biom -m mapping-file-full.alpha.L6index.txt  -f disease_stat -o split-by-disease-state')

get_ipython().system('beta_diversity_through_plots.py -i split-by-disease-state/otu_table.15000.no-diarrhea__disease_stat_IBD__.biom -m mapping-file-full.alpha.L6index.txt -t gg_13_8_otus/trees/97_otus.tree -o split-by-disease-state/beta/15000/ibd --color_by_all_fields -f')

get_ipython().system('beta_diversity_through_plots.py -i split-by-disease-state/otu_table.15000.no-diarrhea__disease_stat_healthy__.biom -m mapping-file-full.alpha.L6index.txt -t gg_13_8_otus/trees/97_otus.tree -o split-by-disease-state/beta/15000/healthy --color_by_all_fields -f')

get_ipython().system('summarize_taxa.py -i otu_table.15000.no-diarrhea.biom -o taxonomic_summaries/no-diarrhea/summaries')

get_ipython().system('make_emperor.py -i beta/15000/unweighted_unifrac_pc.txt -m mapping-file-full.alpha.L6index.txt -t taxonomic_summaries/no-diarrhea/summaries/otu_table.15000.no-diarrhea_L3.txt -o beta/15000/unweighted_unifrac_emperor_pcoa_biplot/ --biplot_fp beta/15000/unweighted_unifrac_emperor_pcoa_biplot/biplot.txt')

get_ipython().system('make_emperor.py -i beta/15000/weighted_unifrac_pc.txt -m mapping-file-full.alpha.L6index.txt -t taxonomic_summaries/no-diarrhea/summaries/otu_table.15000.no-diarrhea_L3.txt -o beta/15000/weighted_unifrac_emperor_pcoa_biplot/ --biplot_fp beta/15000/weighted_unifrac_emperor_pcoa_biplot/biplot.txt')

get_ipython().system('make_emperor.py -i beta/15000/unweighted_unifrac_pc.txt -m mapping-file-full.alpha.L6index.txt -t taxonomic_summaries/no-diarrhea/summaries/otu_table.15000.no-diarrhea_L6.txt -o beta/15000/unweighted_unifrac_emperor_pcoa_biplot-L6/ --biplot_fp beta/15000/unweighted_unifrac_emperor_pcoa_biplot/biplot-L6.txt')

get_ipython().system('make_emperor.py -i beta/15000/weighted_unifrac_pc.txt -m mapping-file-full.alpha.L6index.txt -t taxonomic_summaries/no-diarrhea/summaries/otu_table.15000.no-diarrhea_L6.txt -o beta/15000/weighted_unifrac_emperor_pcoa_biplot-L6/ --biplot_fp beta/15000/weighted_unifrac_emperor_pcoa_biplot/biplot-L6.txt')

Image('beta/15000/screen-shots/unweighted-unifrac-biplot-disease-status.png')

Image('beta/15000/screen-shots/weighted-unifrac-biplot-disease-status.png')

get_ipython().system('compare_categories.py --method permanova -i beta/15000/unweighted_unifrac_dm.txt -m mapping-file-full.alpha.txt -c disease_stat -o beta/15000/stats-unweighted/')

get_ipython().system('compare_categories.py --method permanova -i beta/15000/weighted_unifrac_dm.txt -m mapping-file-full.alpha.txt -c disease_stat -o beta/15000/stats-weighted/')

pd.read_csv('beta/15000/stats-unweighted/permanova_results.txt', sep='\t')

pd.read_csv('beta/15000/stats-weighted/permanova_results.txt', sep='\t')



