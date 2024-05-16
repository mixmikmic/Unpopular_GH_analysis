get_ipython().run_line_magic('run', '_standard_imports.ipynb')

panoptes_previous_metadata_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/pf_6_0_panoptes_samples_final_20170124.txt.gz"
panoptes_final_metadata_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/pf_6_0_panoptes_samples_final_20170206.txt.gz"
sites_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/pf_6_0_sites_20170206.xlsx"
study_summary_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/study_summary_20170206.xlsx"
samples_by_study_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/samples_by_study_20170206.xlsx"
sample_6_0_fn = "/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_sample_metadata.txt"
panoptes_samples_fn = "/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_samples_panoptes_20170206.txt"

tbl_panoptes_samples_previous = (
    etl
    .fromtsv(panoptes_previous_metadata_fn)
    .convertnumbers()
)
print(len(tbl_panoptes_samples_previous.data()))
print(len(tbl_panoptes_samples_previous.distinct('Sample').data()))
tbl_panoptes_samples_previous

tbl_duplicates_ox_code = (
    tbl_panoptes_samples_previous
    .aggregate('IndividualGroup', etl.strjoin(','), 'Sample')
    .rename('value', 'AllSamplesThisIndividual')
)

tbl_duplicates_ox_code.select(lambda rec: len(rec['AllSamplesThisIndividual']) > 10)

tbl_sites = (
    etl.fromxlsx(sites_fn)
    .rename('Source', 'Site_source')
)
print(len(tbl_sites.data()))
print(len(tbl_sites.distinct('Site').data()))
tbl_sites

tbl_panoptes_samples_final = (
    tbl_panoptes_samples_previous
    .join(tbl_duplicates_ox_code, key='IndividualGroup')
    .join(tbl_sites, key='Site')
    .sort(['AlfrescoStudyCode', 'Sample'])
)
print(len(tbl_panoptes_samples_final.data()))
print(len(tbl_panoptes_samples_final.distinct('Sample').data()))
tbl_panoptes_samples_final

# File to send to Olivo
tbl_panoptes_samples_final.totsv(panoptes_final_metadata_fn, lineterminator='\n')

tbl_study_summary = (
    (
        tbl_panoptes_samples_final
        .selecteq('InV5', 'True')
        .valuecounts(
            'AlfrescoStudyCode',
        )
        .rename('count', 'Total in 5.0')
        .cutout('frequency')
    ).outerjoin(
    (
        tbl_panoptes_samples_final
        .valuecounts(
            'AlfrescoStudyCode',
        )
        .rename('count', 'Total in 6.0')
        .cutout('frequency')
    ), key='AlfrescoStudyCode')
    .addfield('New in 6.0', lambda rec: rec['Total in 6.0'] if rec['Total in 5.0'] is None else rec['Total in 6.0'] - rec['Total in 5.0'])
    .outerjoin(
    (
        tbl_panoptes_samples_final
            .selectin('Year', ['-', 'N/A'])
        .valuecounts(
            'AlfrescoStudyCode',
        )
        .rename('count', 'Missing year')
        .cutout('frequency')
    ), key='AlfrescoStudyCode')
    .outerjoin(
    (
        tbl_panoptes_samples_final
            .selectin('Site', ['-'])
        .valuecounts(
            'AlfrescoStudyCode',
        )
        .rename('count', 'Missing site')
        .cutout('frequency')
    ), key='AlfrescoStudyCode')
    .replaceall(None, 0)
    .sort(['AlfrescoStudyCode'])
)
tbl_study_summary.toxlsx(study_summary_fn)
tbl_study_summary.displayall()

partner_columns = ['Sample', 'OxfordSrcCode', 'Site', 'Year', 'AllSamplesThisIndividual']

from pandas import ExcelWriter

studies = tbl_study_summary.values('AlfrescoStudyCode').list()
writer = ExcelWriter(samples_by_study_fn)
for study in studies:
    print(study)
    if tbl_study_summary.selecteq('AlfrescoStudyCode', study).values('Total in 5.0')[0] > 0:
        df = (
            tbl_panoptes_samples_final
            .selecteq('AlfrescoStudyCode', study)
            .selecteq('InV5', 'True')
            .cut(partner_columns)
            .todataframe()
        )
        sheet_name = "%s_old" % study[0:4]
        df.to_excel(writer, sheet_name, index=False)
    if tbl_study_summary.selecteq('AlfrescoStudyCode', study).values('New in 6.0')[0] > 0:
        df = (
            tbl_panoptes_samples_final
            .selecteq('AlfrescoStudyCode', study)
            .selecteq('InV5', 'False')
            .cut(partner_columns)
            .todataframe()
        )
        sheet_name = "%s_new" % study[0:4]
        df.to_excel(writer, sheet_name, index=False)
writer.save()
    

tbl_sample_6_0 = (
    etl
    .fromtsv(sample_6_0_fn)
    .cutout('study')
    .cutout('source_code')
    .cutout('run_accessions')
)
print(len(tbl_sample_6_0.data()))
print(len(tbl_sample_6_0.distinct('sample').data()))
tbl_sample_6_0

tbl_panoptes_samples = (
    tbl_panoptes_samples_final
    .addfield('PreferredSample', lambda rec: rec['DiscardAsDuplicate'] == 'False')
    .convert('Year', lambda v: '', where=lambda r: r['PreferredSample'] == False)
    .convert('Date', lambda v: '', where=lambda r: r['PreferredSample'] == False)
    .cutout('DiscardAsDuplicate')
    .join(tbl_sample_6_0, lkey='Sample', rkey='sample')
    .sort(['AlfrescoStudyCode', 'Sample'])
)
print(len(tbl_panoptes_samples.data()))
print(len(tbl_panoptes_samples.distinct('Sample').data()))
tbl_panoptes_samples

tbl_panoptes_samples.valuecounts('DiscardAsDuplicate')

tbl_panoptes_samples.valuecounts('PreferredSample')

tbl_panoptes_samples.valuecounts('Year').displayall()

tbl_panoptes_samples.totsv(panoptes_samples_fn, lineterminator='\n')



