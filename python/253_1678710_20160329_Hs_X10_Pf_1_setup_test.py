get_ipython().run_line_magic('run', 'standard_imports.ipynb')

input_dir = '/lustre/scratch109/malaria/Hs_X10_Pf_1/input'
output_dir = '/lustre/scratch109/malaria/Hs_X10_Pf_1/output'
test_dir = '/lustre/scratch109/malaria/Hs_X10_Pf_1/test'
get_ipython().system('mkdir -p {input_dir}')
get_ipython().system('mkdir -p {output_dir}')
get_ipython().system('mkdir -p {test_dir}')

lanelets_fn = '/nfs/team112_internal/production_files/Hs/x10/metrics/oxcode_cram.tab'
GF5122_C_irods = "%s/GF5122_C.cram.irods" % test_dir
GF5122_C_fofn = "/nfs/team112_internal/production/release_build/Pf/Hs_X10_Pf_1/Hs_X10_Pf_1.lanelets.fofn"

get_ipython().system('grep GF5122 {lanelets_fn}')

get_ipython().system('grep GF5122 {lanelets_fn} > {GF5122_C_irods}')

get_ipython().system('cat {GF5122_C_irods}')

cwd = get_ipython().getoutput('pwd')
cwd = cwd[0]

get_ipython().run_line_magic('cd', '{test_dir}')

tbl_GF5122_C_lanelets = etl.fromtsv(GF5122_C_irods).pushheader(['sample', 'file'])
for rec in tbl_GF5122_C_lanelets.data():
    get_ipython().system('iget {rec[1]}')

cwd

get_ipython().run_line_magic('cd', '{cwd}')
get_ipython().system('pwd')

tbl_fofn = (tbl_GF5122_C_lanelets
 .sub('sample', '-', '_')
 .sub('file', '/seq/[0-9]+/(.*)', '%s/\\1' % test_dir)
 .rename('file', 'path')
 .cut(['path', 'sample'])
)
tbl_fofn

tbl_fofn.totsv(GF5122_C_fofn)

GF5122_C_fofn







