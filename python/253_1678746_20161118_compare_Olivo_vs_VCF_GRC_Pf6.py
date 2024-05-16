get_ipython().run_line_magic('run', '_standard_imports.ipynb')

output_dir = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161118_compare_Olivo_vs_VCF_GRC_Pf6"
get_ipython().system('mkdir -p {output_dir}')
reads_6_0_results_fn = "/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161117_run_Olivo_GRC/grc/AllCallsBySample.tab"
vcf_6_0_results_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161118_GRC_from_VCF/Pf_6_GRC_from_vcf.xlsx"

all_calls_crosstab_fn = "%s/all_calls_crosstab.xlsx" % output_dir
discordant_calls_crosstab_fn = "%s/discordant_calls_crosstab.xlsx" % output_dir
discordant_nonmissing_calls_crosstab_fn = "%s/discordant_nonmissing_calls_crosstab.xlsx" % output_dir

tbl_read_results = (
    etl
    .fromtsv(reads_6_0_results_fn)
    .distinct('Sample')
    .rename('mdr2_484[P]', 'mdr2_484[T]')
    .rename('fd_193[P]', 'fd_193[D]')
)
print(len(tbl_read_results.data()))
tbl_read_results

tbl_vcf_results = (
    etl
    .fromxlsx(vcf_6_0_results_fn)
)
print(len(tbl_vcf_results.data()))
tbl_vcf_results

tbl_both_results = (
    tbl_read_results
    .cutout('Num')
    .join(tbl_vcf_results.replaceall('.', '-'), key='Sample', lprefix='reads_', rprefix='vcf_')
    .convertall(lambda x: ','.join(sorted(x.split(','))))
)
print(len(tbl_both_results.data()))

df_both_results = tbl_both_results.todataframe()

loci = list(tbl_read_results.header()[2:])
print(len(loci))

writer = pd.ExcelWriter(all_calls_crosstab_fn)
for i in np.arange(1, 30):
    pd.crosstab(
        df_both_results.ix[:, i],
        df_both_results.ix[:, i+29],
        margins=True
    ).to_excel(writer, loci[i-1].split('[')[0]) # Excel doesn't like the [CMNVK] like endings
writer.save()

writer = pd.ExcelWriter(discordant_calls_crosstab_fn)
for i in np.arange(1, 30):
    pd.crosstab(
        df_both_results.ix[:, i][df_both_results.ix[:, i].str.upper() != df_both_results.ix[:, i+29].str.upper()],
        df_both_results.ix[:, i+29][df_both_results.ix[:, i].str.upper() != df_both_results.ix[:, i+29].str.upper()],
        margins=True
    ).to_excel(writer, loci[i-1].split('[')[0])
writer.save()

writer = pd.ExcelWriter(discordant_nonmissing_calls_crosstab_fn)
for i in np.arange(1, 30):
    pd.crosstab(
        df_both_results.ix[:, i][
            (df_both_results.ix[:, i].str.upper() != df_both_results.ix[:, i+29].str.upper()) &
            (df_both_results.ix[:, i] != '-') &
            (df_both_results.ix[:, i+29] != '-')
        ],
        df_both_results.ix[:, i+29][
            (df_both_results.ix[:, i].str.upper() != df_both_results.ix[:, i+29].str.upper()) &
            (df_both_results.ix[:, i] != '-') &
            (df_both_results.ix[:, i+29] != '-')
        ],
        margins=True
    ).to_excel(writer, loci[i-1].split('[')[0])
writer.save()



# Het call made at final SNP despite only one read
# Unclear which result really correct
df_both_results[
    (df_both_results['reads_crt_72-76[CVMNK]'] == 'CVIDT,CVIET') &
    (df_both_results['vcf_crt_72-76[CVMNK]'] == 'CVIDK,CVIET')]

# Many variants here had small numbers of ALT reads but were called het
# Reads results probably correct
df_both_results[
    (df_both_results['reads_crt_72-76[CVMNK]'] == 'CVIET,CVMNK,SVMNT') &
    (df_both_results['vcf_crt_72-76[CVMNK]'] == 'CVMNK,CVMNT')]

# Get wrong call here becuase middle GT is 0/0. Would get correct call if we used PGT call which is 1|0
# Reads results probably correct
df_both_results[(df_both_results['reads_dhps_436[S]'] == 'A,F') & (df_both_results['vcf_dhps_436[S]'] == 'A,S')]

# For all of the following, final variant looks like a het (always 2+ reads) but called hom
# Reads results probably correct for all of these
df_both_results[(df_both_results['reads_dhps_540[K]'] == 'E,N') & (df_both_results['vcf_dhps_540[K]'] == 'E,K')]





