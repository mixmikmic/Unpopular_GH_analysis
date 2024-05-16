get_ipython().run_line_magic('run', '_standard_imports.ipynb')

results_dir = '/nfs/team112_internal/rp7/data/methods-dev/analysis/20161005_Pf_60_failures'
get_ipython().system('mkdir {results_dir}')

jim_fn = '%s/pf_60_fails_studies.tab' % results_dir
results_fn = '%s/pf_60_failures_summary.xlsx' % results_dir
pf3k_vcf_fn = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/SNP_INDEL_Pf3D7_01_v3.combined.filtered.vcf.gz'

# vcf_reader = vcf.Reader(open(pf3k_vcf_fn, 'rz'))
vcf_reader = vcf.Reader(filename=pf3k_vcf_fn)
pf3k_samples = vcf_reader.samples

samples_line = get_ipython().getoutput("zcat /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/SNP_INDEL_Pf3D7_01_v3.combined.filtered.vcf.gz | head -n 500 | grep '#CHROM'")

pf3k_samples = samples_line[0].split('\t')[9:]
print(len(pf3k_samples))
pf3k_samples[0:10]

tbl_failed_samples = (
    etl
    .fromtsv(jim_fn)
    .pushheader(['study', 'ox_code', 'in_pf_5.1'])
    .convert('in_pf_5.1', lambda x: bool(int(x)))
    .addfield('in_pf3k', lambda rec: rec[1] in pf3k_samples)
)
print(len(tbl_failed_samples.data()))
tbl_failed_samples.displayall()

tbl_failed_sample_summary = (
    tbl_failed_samples
#     .valuecounts('study', 'in_pf_5.1', 'in_pf3k')
    .valuecounts('study', 'in_pf_5.1')
    .cutout('frequency')
    .rename('count', 'Number of samples')
#     .sort(('study', 'in_pf_5.1', 'in_pf3k'))
    .sort(('study', 'in_pf_5.1'))
)

tbl_failed_sample_summary.displayall()

tbl_failed_sample_summary.toxlsx(results_fn)



