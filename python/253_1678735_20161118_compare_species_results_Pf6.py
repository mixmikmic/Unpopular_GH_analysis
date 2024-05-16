get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')

output_dir = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161118_compare_species_results_Pf6"
get_ipython().system('mkdir -p {output_dir}')
olivo_7979_results_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161118_compare_7979_vs_Pf6_GRC/samplesMeta5x-V1.0.xlsx"
# reads_6_0_results_fn = "/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161117_run_Olivo_GRC/grc/AllCallsBySample.tab"
# vcf_6_0_results_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161115_GRC_from_VCF/Pf_6_GRC_from_vcf.xlsx"

all_calls_crosstab_fn = "%s/all_calls_crosstab.xlsx" % output_dir
discordant_calls_crosstab_fn = "%s/discordant_calls_crosstab.xlsx" % output_dir
discordant_nonmissing_calls_crosstab_fn = "%s/discordant_nonmissing_calls_crosstab.xlsx" % output_dir

speciator_pv_unique_coverage_fn = "/nfs/team112_internal/rp7/data/pv/analysis/20161003_pv_3_0_sample_metadata/pv_unique_coverage.txt"
reads_6_0_species_results_fn = "/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161117_run_Olivo_GRC/species/AllSamples-AllTargets.classes.tab"

sample_swaps_51_60 = [
    'PP0011-C', 'PP0010-C', 'PC0003-0', 'PH0027-C', 'PH0027-C', 'PG0313-C', 'PG0305-C', 'PG0304-C', 'PG0282-C',
    'PG0309-C', 'PG0334-C', 'PG0306-C', 'PG0332-C', 'PG0311-C', 'PG0330-C', 'PG0310-C', 'PG0312-C', 'PG0335-C',
    'PG0280-C', 'PG0281-C', 'PG0308-C', 'PP0010-C', 'PP0011-C', 'PC0003-0', 'PH0027-C', 'PH0027-C', 'PG0335-C',
    'PG0308-C', 'PG0332-C', 'PG0330-C', 'PG0311-C', 'PG0305-C', 'PG0309-C', 'PG0304-C', 'PG0313-C', 'PG0280-C',
    'PG0312-C', 'PG0334-C', 'PG0306-C', 'PG0281-C', 'PG0282-C', 'PG0310-C'
]

tbl_5_1_species = (
    etl
    .fromxlsx(olivo_7979_results_fn)
)
print(len(tbl_5_1_species.data()))
tbl_5_1_species

tbl_6_0_species = (
    etl
    .fromtsv(reads_6_0_species_results_fn)
    .convertnumbers()
)
print(len(tbl_6_0_species.data()))
tbl_6_0_species

aggregation = collections.OrderedDict()
aggregation['count'] = len
aggregation['sum'] = 'pv_unique_coverage', sum

tbl_speciator_pv_unique_coverage = (
    etl
    .fromtsv(speciator_pv_unique_coverage_fn)
    .convertnumbers()
    .aggregate('ox_code', aggregation)
    .addfield('pv_unique_coverage', lambda rec: rec['sum'] / rec['count'])
)
print(len(tbl_speciator_pv_unique_coverage.data()))
tbl_speciator_pv_unique_coverage

len(tbl_speciator_pv_unique_coverage.duplicates('ox_code'))

tbl_species_5_1_vs_6_0 = (
    tbl_5_1_species
    .join(tbl_6_0_species, key='Sample')
    .cut(['Sample', 'Species', 'SampleClass'])
    .selectnotin('Sample', sample_swaps_51_60)
)
print(len(tbl_species_5_1_vs_6_0.data()))
tbl_species_5_1_vs_6_0

df_species_5_1_vs_6_0 = tbl_species_5_1_vs_6_0.todataframe()

writer = pd.ExcelWriter(all_calls_crosstab_fn)
df_species_5_1_vs_6_0
pd.crosstab(
    df_species_5_1_vs_6_0.ix[:, 1],
    df_species_5_1_vs_6_0.ix[:, 2],
    margins=True
).to_excel(writer, 'All')
pd.crosstab(
    df_species_5_1_vs_6_0.ix[:, 1][df_species_5_1_vs_6_0.ix[:, 1] != df_species_5_1_vs_6_0.ix[:, 2]],
    df_species_5_1_vs_6_0.ix[:, 2][df_species_5_1_vs_6_0.ix[:, 1] != df_species_5_1_vs_6_0.ix[:, 2]],
    margins=True
).to_excel(writer, 'Discordant')
pd.crosstab(
    df_species_5_1_vs_6_0.ix[:, 1][
        (df_species_5_1_vs_6_0.ix[:, 1] != df_species_5_1_vs_6_0.ix[:, 2]) &
        (df_species_5_1_vs_6_0.ix[:, 1] != '-') &
        (df_species_5_1_vs_6_0.ix[:, 2] != '-')
    ],
    df_species_5_1_vs_6_0.ix[:, 2][
        (df_species_5_1_vs_6_0.ix[:, 1] != df_species_5_1_vs_6_0.ix[:, 2]) &
        (df_species_5_1_vs_6_0.ix[:, 1] != '-') &
        (df_species_5_1_vs_6_0.ix[:, 2] != '-')
    ],
    margins=True
).to_excel(writer, 'Non_missing')
writer.save()

df_species_5_1_vs_6_0[
    (df_species_5_1_vs_6_0['Species'] == 'Pf,Pv') &
    (df_species_5_1_vs_6_0['SampleClass'] == 'Pv')
]

tbl_speciator_pv_unique_coverage.selecteq('ox_code', 'PV0025-C')

get_ipython().system('cat /nfs/team112_internal/production_files/Pf/6_0/speciator/PV0025_C_5371_1_nonhuman.tab')

df_species_5_1_vs_6_0[
    (df_species_5_1_vs_6_0['Species'] == 'Pf,Pv') &
    (df_species_5_1_vs_6_0['SampleClass'] == 'Pf*,Pv')
]

get_ipython().system('cat /nfs/team112_internal/production_files/Pf/6_0/speciator/PN0157_C_12483_3_32.tab')

get_ipython().system('cat /nfs/team112_internal/production_files/Pf/6_0/speciator/PV0255_C_8129_2_20.tab')

tbl_species_6_0_vs_speciator = (
    tbl_6_0_species
    .join(tbl_speciator_pv_unique_coverage, lkey='Sample', rkey='ox_code')
    .cut(['Sample', 'SampleClass', 'pv_unique_coverage'])
    .addfield('Pv classification', lambda rec: 'Consensus' if 'Pv' in rec['SampleClass'] and not 'Pv*' in rec['SampleClass']
        else 'Evidence' if 'Pv*' in rec['SampleClass']
        else 'Unknown' if rec['SampleClass'] == '-'
        else 'No Pv'
    )
)
print(len(tbl_species_6_0_vs_speciator.data()))
tbl_species_6_0_vs_speciator

tbl_species_6_0_vs_speciator.valuecounts('Pv classification').displayall()

tbl_species_6_0_vs_speciator.selectgt('pv_unique_coverage', 0.0).valuecounts('Pv classification').displayall()

tbl_species_6_0_vs_speciator.selecteq('pv_unique_coverage', 0.0).valuecounts('Pv classification').displayall()

tbl_species_6_0_vs_speciator.valuecounts('Pv classification').toxlsx("%s/Pv_classification.xlsx" % output_dir)

# Note I have removed samples with zero Pv coverage as 
df_species_6_0_vs_speciator = tbl_species_6_0_vs_speciator.todataframe()
df_species_6_0_vs_speciator = df_species_6_0_vs_speciator[df_species_6_0_vs_speciator['pv_unique_coverage'] > 0]
df_species_6_0_vs_speciator['Log10(Pv coverage)'] = np.log10(df_species_6_0_vs_speciator['pv_unique_coverage'])
# df_species_6_0_vs_speciator = df_species_6_0_vs_speciator[np.logical_not(np.isinf(df_species_6_0_vs_speciator['Log10(Pv coverage)']))]
# df_species_6_0_vs_speciator['Log10(Pv coverage)'][np.isinf(df_species_6_0_vs_speciator['Log10(Pv coverage)'])] = -5.0

df_species_6_0_vs_speciator['Log10(Pv coverage)'].describe()

ax = sns.violinplot(x="Pv classification", y="Log10(Pv coverage)", data=df_species_6_0_vs_speciator)

fig=plt.figure(figsize=(6,4))
ax=fig.add_subplot(1, 1, 1)
ax = sns.boxplot(x="Pv classification", y="Log10(Pv coverage)", data=df_species_6_0_vs_speciator, ax=ax)
# ax = sns.swarmplot(x="Pv classification", y="Log10(Pv coverage)", data=df_species_6_0_vs_speciator,
#                    color="white", edgecolor="gray")
ax.set_xticklabels(['No Pv\nn=7,595', 'Unknown\nn=31', 'Evidence\nn=106', 'Consensus\nn=139'])
fig.savefig("%s/Pv_in_Pf.png" % output_dir, dpi=300)

tbl_species_6_0_vs_speciator.selecteq('Pv classification', 'No Pv').selectgt('pv_unique_coverage', 0.3).displayall()

np.log10(0.507)

tbl_species_6_0_vs_speciator

# This is the threshold for inclusion in Pv build
tbl_species_6_0_vs_speciator.selectgt('pv_unique_coverage', 5.0).valuecounts('Pv classification').displayall()

# The following are inlcuded in Pv 3.0 build, but are not given a species
# The following few cells look at some common variants in Pf 6.0, to see if they affect anchors in any of these
# samples, but they don't
tbl_species_6_0_vs_speciator.selectgt('pv_unique_coverage', 5.0).selecteq('Pv classification', 'Unknown')

vcf_file_format = "/nfs/team112_internal/production_files/Pf/6_0/vcf/SNP_INDEL_%s.combined.filtered.vcf.gz"
def non_ref_sample(chrom='Pf_M76611', pos=960):
    vcf_reader = vcf.Reader(filename=vcf_file_format % chrom)
    samples = vcf_reader.samples
    for record in vcf_reader.fetch(chrom, pos-1, pos):
        print(record, record.FILTER)
        for sample in samples:
            GT = record.genotype(sample)['GT']
            if not GT in ['0/0', './.']:
                print(sample, record.genotype(sample))

non_ref_sample()

non_ref_sample(pos=1033)

non_ref_sample(pos=1100)

# Note that was run before I recreated the table to have mean coverage for each ox_code
tbl_species_6_0_vs_speciator.selecteq('Sample', 'PH1186-C')

tbl_species_6_0_vs_speciator.selecteq('pv_unique_coverage', 0.0).valuecounts('Pv classification').displayall()

(
    etl
    .fromtsv(speciator_pv_unique_coverage_fn)
    .convertnumbers()
    .selecteq('ox_code', 'PH1186-C')
)

get_ipython().system('cat /nfs/team112_internal/production_files/Pf/6_0/speciator/PH1186_C_13253_5_7.tab')

get_ipython().system('cat /nfs/team112_internal/production_files/Pf/6_0/speciator/PH1186_C_14323_2_6.tab')



pv_unique_coverage = collections.OrderedDict()
pv_unique_coverage['Pv consensus'] = (
    tbl_species_6_0_vs_speciator
    .select(lambda rec: 'Pv' in rec['SampleClass'] and not 'Pv*' in rec['SampleClass'])
    .values('pv_unique_coverage')
    .array()
)
pv_unique_coverage['Pv evidence'] = (
    tbl_species_6_0_vs_speciator
    .select(lambda rec: 'Pv*' in rec['SampleClass'])
    .values('pv_unique_coverage')
    .array()
)
pv_unique_coverage['No Pv'] = (
    tbl_species_6_0_vs_speciator
    .select(lambda rec: not 'Pv' in rec['SampleClass'])
    .values('pv_unique_coverage')
    .array()
)



