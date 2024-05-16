get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')
import pandas as pd

output_dir = '/nfs/team112_internal/rp7/data/methods-dev/pf3k_techbm/20170216_core_defintion_comparisons'
gff_fn = '/lustre/scratch118/malaria/team112/pipelines/resources/pf3k_methods/resources/snpEff/data/Pfalciparum_GeneDB_Aug2015/genes.gff'

get_ipython().system("grep 'product=term%3Dstevor%3B' {gff_fn}")

get_ipython().system("grep 'Name=VAR' {gff_fn}")



