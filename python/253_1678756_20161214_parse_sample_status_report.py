get_ipython().run_line_magic('run', '_standard_imports.ipynb')

output_dir = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_parse_sample_status_report"
get_ipython().system('mkdir -p {output_dir}')

sample_status_report_fn = "%s/2016_12_07_report_sample_status.xls" % output_dir
output_fn = "%s/2016_12_07_report_sample_status_pf_pv.xlsx" % output_dir
output_fn = "%s/2016_12_07_report_sample_status.txt" % output_dir

tbl_studies = (
    etl
    .fromxls(sample_status_report_fn, 'Report')
    .skip(2)
    .selectin('Taxon', ['PF', 'PV'])
    .selectne('Project Code', '')
)
project_codes = tbl_studies.values('Project Code').array()
alfresco_codes = tbl_studies.values('Alfresco Code').array()
print(len(tbl_studies.data()))
tbl_studies.tail()

project_codes

tbl_sample_status_report = (
    etl
    .fromxls(sample_status_report_fn, project_codes[0])
    .skip(10)
    .selectne('Oxford Code', '')
    .addfield('study', alfresco_codes[0])
)
for i, project_code in enumerate(project_codes):
    if i > 0:
        tbl_sample_status_report = (
            tbl_sample_status_report
            .cat(
                etl
                .fromxls(sample_status_report_fn, project_code)
                .skip(10)
                .selectne('Oxford Code', '')
                .addfield('study', alfresco_codes[i])
            )
        )

tbl_sample_status_report

len(tbl_sample_status_report.data())

tbl_sample_status_report.totsv(output_fn, lineterminator='\n')

tbl_sample_status_reloaded = etl.fromtsv(output_fn)
print(len(tbl_sample_status_reloaded.data()))
tbl_sample_status_reloaded

output_fn

tbl_sample_status_reloaded.tail()

tbl_sample_status_reloaded.duplicates('Oxford Code')

tbl_sample_status_reloaded.valuecounts('study').displayall()

tbl_sample_status_reloaded.valuecounts('Country of Origin').displayall()

tbl_sample_status_reloaded.selecteq('Country of Origin', '').displayall()



