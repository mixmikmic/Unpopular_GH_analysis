get_ipython().run_line_magic('run', 'standard_imports.ipynb')

pf_51_manifest_fn = '/nfs/team112_internal/production/release_build/Pf/5_1_release_packages/pf_51_freeze_manifest.tab'
WillH_1_samples_fn = '/lustre/scratch109/malaria/WillH_1/meta/Antoine_samples_vrpipe2.txt'

tbl_pf_51_manifest = etl.fromtsv(pf_51_manifest_fn)
tbl_pf_51_manifest

tbl_WillH_1_samples = etl.fromtsv(WillH_1_samples_fn).pushheader(['sample'])
tbl_WillH_1_samples

tbl_vrpipe = (tbl_WillH_1_samples
    .join(tbl_pf_51_manifest, key='sample')
    .cut(['path', 'sample'])
)
print(len(tbl_vrpipe.data()))
tbl_vrpipe

tbl_missing = (tbl_WillH_1_samples
    .antijoin(tbl_pf_51_manifest, key='sample')
)
print(len(tbl_missing.data()))
tbl_missing.displayall()



