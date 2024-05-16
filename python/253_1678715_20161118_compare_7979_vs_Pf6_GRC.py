get_ipython().run_line_magic('run', '_standard_imports.ipynb')

output_dir = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161118_compare_7979_vs_Pf6_GRC"
get_ipython().system('mkdir -p {output_dir}')
olivo_7979_results_fn = "%s/samplesMeta5x-V1.0.xlsx" % output_dir
richard_6_0_results_fn = "/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161117_run_Olivo_GRC/grc/AllCallsBySample.tab"

all_calls_crosstab_fn = "%s/Pf6_vs_7979_all_calls_crosstab.xlsx" % output_dir
discordant_calls_crosstab_fn = "%s/Pf6_vs_7979_discordant_calls_crosstab.xlsx" % output_dir
discordant_nonmissing_calls_crosstab_fn = "%s/Pf6_vs_7979_discordant_nonmissing_calls_crosstab.xlsx" % output_dir

bwa_aln_fofn = '/nfs/team112_internal/rp7/data/Pf/hrp/metadata/hrp_manifest_20160621.txt'
bwa_mem_fofn = '/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161117_run_Olivo_GRC/pf_60_mergelanes.txt'

get_ipython().system('wc -l {bwa_aln_fofn}')

get_ipython().system('wc -l {bwa_mem_fofn}')

tbl_bwa_aln = (
    etl
    .fromxlsx(olivo_7979_results_fn)
)
print(len(tbl_bwa_aln.data()))
tbl_bwa_aln

tbl_bwa_mem = (
    etl
    .fromtsv(richard_6_0_results_fn)
    .rename('mdr2_484[P]', 'mdr2_484[T]')
    .rename('fd_193[P]', 'fd_193[D]')
)
print(len(tbl_bwa_mem.data()))
tbl_bwa_mem

loci = list(tbl_bwa_mem.header()[2:])
print(len(loci))

tbl_both_results = (
    tbl_bwa_aln
    .cut(['Sample'] + loci)
    .join(tbl_bwa_mem.cut(['Sample'] + loci), key='Sample', lprefix='bwa_aln_', rprefix='bwa_mem_')
)
print(len(tbl_both_results.data()))

tbl_both_results

df_both_results = tbl_both_results.todataframe()

writer = pd.ExcelWriter(all_calls_crosstab_fn)
for i in np.arange(1, 30):
    pd.crosstab(
        df_both_results.ix[:, i],
        df_both_results.ix[:, i+29],
        margins=True
    ).to_excel(writer, loci[i-1].split('[')[0]) # Excel doesn't like the [CMNVK] endings
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



tbl_both_results.selecteq(16, 'S').selecteq(16+29, 'N')

tbl_both_results.selecteq(16, 'N').selecteq(16+29, 'S')

tbl_both_results.selecteq(19, 'G').selecteq(19+29, 'A')

tbl_both_results.selecteq(19, 'A').selecteq(19+29, 'G')

tbl_both_results.selecteq(24, 'F').selecteq(24+29, 'Y')

tbl_both_results.selecteq(24, 'Y').selecteq(24+29, 'F')

# See methods-dev/notebooks/20160621_HRP_sample_metadata.ipynb
fofns = collections.OrderedDict()

fofns['bwa_aln'] = bwa_aln_fofn
fofns['bwa_mem'] = bwa_mem_fofn
fofns['pf_community_5_1'] = '/nfs/team112_internal/production_files/Pf/5_1/pf_51_samplebam_cleaned.fofn'
fofns['pf_community_5_0'] = '/nfs/team112_internal/production/release_build/Pf/5_0_release_packages/pf_50_freeze_manifest_nolab_olivo.tab'
fofns['pf3k_pilot_5_0_broad'] = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/pf3k_metadata.tab'
fofns['pdna'] = '/nfs/team112_internal/production_files/Pf/PDNA/pf_pdna_new_samplebam.fofn'
fofns['conway'] = '/nfs/team112_internal/production_files/Pf/1147_Conway/pf_conway_metadata.fofn'
fofns['trac'] = '/nfs/team112_internal/rp7/data/Pf/hrp/fofns/olivo_TRAC.fofn'
fofns['fanello'] = '/nfs/team112_internal/rp7/data/Pf/hrp/fofns/olivo_fanello.fofn'

for fofn in fofns:
    print(fofn)
    get_ipython().system('grep PG0282 {fofns[fofn]}')

def show_rg(sample='PG0282'):
    line = get_ipython().getoutput('grep {sample} {bwa_aln_fofn}')
    bam_fn = line[0].split('\t')[0]
    rg = get_ipython().getoutput('samtools view -H {bam_fn} | grep RG')
    print(rg[0])
    line = get_ipython().getoutput('grep {sample} {bwa_mem_fofn}')
    bam_fn = line[0].split('\t')[0]
    rg = get_ipython().getoutput('samtools view -H {bam_fn} | grep RG')
    print(rg[0])

for sample_id in ['PG0282-C', 'PG0304-C', 'PG0312-C', 'PG0313-C', 'PG0330-C', 'PG0332-C', 'PG0334-C', 'PG0335-C']:
    print(sample_id)
    show_rg(sample_id)



