get_ipython().run_line_magic('run', '_standard_imports.ipynb')

output_dir = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release"
get_ipython().system('mkdir -p {output_dir}')

report_sample_status_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_parse_sample_status_report/2016_12_07_report_sample_status.txt"
solaris_fn = "%s/PF_metadata_base.csv" % output_dir
olivo_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161118_compare_7979_vs_Pf6_GRC/samplesMeta5x-V1.0.xlsx"
sample_5_0_fn = "/nfs/team112_internal/production/release_build/5_0_study_samples.tab"
sample_5_1_fn = "/nfs/team112_internal/production/release_build/5_1_study_samples_all.tab"
sample_6_0_fn = "/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_sample_metadata.txt"

study_summary_fn = "%s/Pf_6_0_study_summary.xlsx" % output_dir

tbl_report_sample_status = (
    etl
    .fromtsv(report_sample_status_fn)
    .selecteq('Taxon', 'PF')
)
print(len(tbl_report_sample_status.data()))
tbl_report_sample_status

tbl_report_sample_status.valuecounts('State').displayall()

tbl_report_sample_status.valuecounts('Note').displayall()

tbl_solaris = (etl
    .fromcsv(solaris_fn, encoding='latin1')
    .cut(['oxford_code', 'type', 'country', 'sampling_date'])
#     .rename('sampling_date', 'cinzia_sampling_date')
    .unique('oxford_code')
)
print(len(tbl_solaris.data()))
tbl_solaris.tail()

tbl_olivo = (etl
    .fromxlsx(olivo_fn)
)
print(len(tbl_olivo.data()))
tbl_olivo.tail()

tbl_sample_5_0 = (etl
    .fromtsv(sample_5_0_fn)
    .pushheader(['study', 'oxford_code'])
)
print(len(tbl_sample_5_0.data()))
tbl_sample_5_0.tail()
samples_5_0 = tbl_sample_5_0.values('oxford_code').array()
print(len(samples_5_0))

tbl_sample_5_1 = (etl
    .fromtsv(sample_5_1_fn)
    .pushheader(['study', 'oxford_code'])
)
print(len(tbl_sample_5_1.data()))
tbl_sample_5_1.tail()
samples_5_1 = tbl_sample_5_1.values('oxford_code').array()
print(len(samples_5_1))

tbl_sample_6_0 = (etl
    .fromtsv(sample_6_0_fn)
)
print(len(tbl_sample_6_0.data()))
samples_6_0 = tbl_sample_6_0.values('sample').array()
print(len(samples_6_0))
tbl_sample_6_0.tail()

def which_release(rec):
    if rec['in_5_0']:
        if (not rec['in_5_1']) or (not rec['in_6_0']):
            return('5_0_NOT_SUBSEQUENT')
        else:
            return('5_0')
    elif rec['in_5_1']:
        if (not rec['in_6_0']):
            return('5_1_NOT_SUBSEQUENT')
        else:
            return('5_1')
    elif rec['in_6_0']:
        return('6_0')
    elif rec['State'] == 'in progress':
        return('in_progress')
    else:
        return('no_release')

tbl_metadata = (
    tbl_report_sample_status
    .leftjoin(tbl_solaris, lkey='Oxford Code', rkey='oxford_code')
    .addfield('in_5_0', lambda rec: rec['Oxford Code'] in samples_5_0)
    .addfield('in_5_1', lambda rec: rec['Oxford Code'] in samples_5_1)
    .addfield('in_6_0', lambda rec: rec['Oxford Code'] in samples_6_0)
    .addfield('release', which_release)
    .leftjoin(tbl_olivo, lkey='Oxford Code', rkey='Sample')
    .outerjoin(tbl_sample_6_0.rename('study', 'study2'), lkey='Oxford Code', rkey='sample')
)
print(len(tbl_metadata.data()))
tbl_metadata.tail()

tbl_metadata.valuecounts('Country of Origin', 'country').displayall()

tbl_sample_6_0.antijoin(tbl_report_sample_status, rkey='Oxford Code', lkey='sample')

tbl_metadata.valuecounts('release').displayall()

tbl_metadata.pivot('study', 'release', 'release', len).displayall()

tbl_metadata.pivot('study', 'release', 'release', len).toxlsx(study_summary_fn)

tbl_panoptes_samples = (
    tbl_metadata
    .selectnotnone('study2')
)
print(len(tbl_panoptes_samples.data()))

