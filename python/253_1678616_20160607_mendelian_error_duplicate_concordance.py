get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')

release5_final_files_dir = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0'
# chrom_vcf_fn = "%s/SNP_INDEL_Pf3D7_14_v3.combined.filtered.vcf.gz" % (release5_final_files_dir)
chrom_vcf_fn = "%s/SNP_INDEL_WG.combined.filtered.crosses.vcf.gz" % (release5_final_files_dir)

output_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160607_mendelian_error_duplicate_concordance"
get_ipython().system('mkdir -p {output_dir}/vcf')
chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_Pf3D7_14_v3.analysis.vcf.gz" % output_dir

release5_crosses_metadata_txt_fn = '../../meta/pf3k_release_5_crosses_metadata.txt'

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'

vfp_tool_configs = collections.OrderedDict()
vfp_tool_configs['5_2'] = '/nfs/team112/software/htslib/vfp/just_call.config'
vfp_tool_configs['6_3'] = '%s/call_3het.config' % output_dir

get_ipython().run_cell_magic('writefile', "{vfp_tool_configs['6_3']}", 'testing=0\nfilters=gtcall\ninput_is_vcf=1\noutput_is_vcf=1\ncallMinCov=6\ncallMinAlleleCov=3\n')

chrom_vcf_fn

tbl_release5_crosses_metadata = etl.fromtsv(release5_crosses_metadata_txt_fn)
print(len(tbl_release5_crosses_metadata.data()))
tbl_release5_crosses_metadata

all_samples = ','.join(tbl_release5_crosses_metadata.values('sample'))

all_samples

tbl_release5_crosses_metadata.duplicates('clone').addrownumbers().select(lambda rec: rec['row']%2 == 1).values('sample').list()

tbl_release5_crosses_metadata.duplicates('clone').addrownumbers().select(lambda rec: rec['row']%2 == 0).values('sample').list()

replicates_first = [
 'PG0112-C',
 'PG0062-C',
 'PG0053-C',
 'PG0053-C',
 'PG0053-C',
 'PG0055-C',
 'PG0055-C',
 'PG0056-C',
 'PG0004-CW',
 'PG0111-C',
 'PG0100-C',
 'PG0079-C',
 'PG0104-C',
 'PG0086-C',
 'PG0095-C',
 'PG0078-C',
 'PG0105-C',
 'PG0102-C']

replicates_second = [
 'PG0112-CW',
 'PG0065-C',
 'PG0055-C',
 'PG0056-C',
 'PG0067-C',
 'PG0056-C',
 'PG0067-C',
 'PG0067-C',
 'PG0052-C',
 'PG0111-CW',
 'PG0100-CW',
 'PG0079-CW',
 'PG0104-CW',
 'PG0086-CW',
 'PG0095-CW',
 'PG0078-CW',
 'PG0105-CW',
 'PG0102-CW']

quads_first = [
 'PG0053-C',
 'PG0053-C',
 'PG0053-C',
 'PG0055-C',
 'PG0055-C',
 'PG0056-C']

quads_second = [
 'PG0055-C',
 'PG0056-C',
 'PG0067-C',
 'PG0056-C',
 'PG0067-C',
 'PG0067-C']

np.in1d(replicates_first, tbl_release5_crosses_metadata.values('sample').array())

rep_index_first = np.in1d(tbl_release5_crosses_metadata.values('sample').array(), replicates_first)
rep_index_second = np.in1d(tbl_release5_crosses_metadata.values('sample').array(), replicates_second)
print(np.sum(rep_index_first))
print(np.sum(rep_index_second))

sample_ids = tbl_release5_crosses_metadata.values('sample').array()

rep_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_first)]
rep_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_second)]
print(len(rep_index_first))
print(rep_index_first)
print(len(rep_index_second))
print(rep_index_second)

sample_ids = tbl_release5_crosses_metadata.values('sample').array()

quad_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_first)]
quad_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_second)]
print(len(quad_index_first))
print(quad_index_first)
print(len(quad_index_second))
print(quad_index_second)

tbl_release5_crosses_metadata.duplicates('clone').displayall()

tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array()

(tbl_release5_crosses_metadata
 .selecteq('study_title', '3D7xHB3 cross progeny')
 .selecteq('parent_or_progeny', 'parent')
 .values('sample')
 .array()
)

def create_variants_npy(vcf_fn):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.variants(
        vcf_fn,
        dtypes={
            'REF':                      'a10',
            'ALT':                      'a10',
            'RegionType':               'a25',
            'VariantType':              'a40',
            'RU':                       'a40',
            'set':                      'a40',
            'SNPEFF_AMINO_ACID_CHANGE': 'a20',
            'SNPEFF_CODON_CHANGE':      'a20',
            'SNPEFF_EFFECT':            'a33',
            'SNPEFF_EXON_ID':            'a2',
            'SNPEFF_FUNCTIONAL_CLASS':   'a8',
            'SNPEFF_GENE_BIOTYPE':      'a14',
            'SNPEFF_GENE_NAME':         'a20',
            'SNPEFF_IMPACT':             'a8',
            'SNPEFF_TRANSCRIPT_ID':     'a20',
            'culprit':                  'a14',
        },
        arities={
            'ALT':   1,
            'AF':    1,
            'AC':    1,
            'MLEAF': 1,
            'MLEAC': 1,
            'RPA':   2,
            'ANN':   1,
        },
        fills={
            'VQSLOD': np.nan,
            'QD': np.nan,
            'MQ': np.nan,
            'MQRankSum': np.nan,
            'ReadPosRankSum': np.nan,
            'FS': np.nan,
            'SOR': np.nan,
            'DP': np.nan,
        },
        flatten_filter=True,
        verbose=False,
        cache=True,
        cachedir=output_dir
    )

def create_calldata_npy(vcf_fn, max_alleles=2):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.calldata_2d(
        vcf_fn,
        fields=['GT', 'AD'],
        dtypes={
            'AD': 'u2',
        },
        arities={
            'AD': max_alleles,
        },
        verbose=False,
        cache=True,
        cachedir=output_dir
    )

def create_analysis_vcf(input_vcf_fn=chrom_vcf_fn, region='Pf3D7_14_v3:1000000-1100000', vfp_tool_config='5_2',
                        output_vcf_fn=None, BCFTOOLS=BCFTOOLS, rewrite=False):
    if output_vcf_fn is None:
        output_vcf_fn = "%s/vcf/SNP_INDEL_%s_%s.analysis.vcf.gz" % (output_dir, region, vfp_tool_config)
    intermediate_fns = collections.OrderedDict()
    for intermediate_file in ['biallelic', 'regenotyped', 'new_af', 'nonref', 'pass', 'minimal', 'analysis', 'SNP', 'INDEL',
                             'SNP_BIALLELIC', 'SNP_MULTIALLELIC', 'SNP_MIXED', 'INDEL_BIALLELIC', 'INDEL_MULTIALLELIC',
                             'gatk_new_af', 'gatk_nonref', 'gatk_pass', 'gatk_minimal', 'gatk_analysis', 'gatk_SNP', 'gatk_INDEL',
                             'gatk_SNP_BIALLELIC', 'gatk_SNP_MULTIALLELIC', 'gatk_SNP_MIXED', 'gatk_INDEL_BIALLELIC', 'gatk_INDEL_MULTIALLELIC']:
        intermediate_fns[intermediate_file] = output_vcf_fn.replace('analysis', intermediate_file)

#     if not os.path.exists(subset_vcf_fn):
#         !{BCFTOOLS} view -Oz -o {subset_vcf_fn} -s {validation_samples} {chrom_vcf_fn}
#         !{BCFTOOLS} index --tbi {subset_vcf_fn}

    if rewrite or not os.path.exists(intermediate_fns['biallelic']):
        if region is not None:
            get_ipython().system("{BCFTOOLS} annotate --regions {region} --remove FORMAT/DP,FORMAT/GQ,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} norm -m -any -Oz -o {intermediate_fns['biallelic']}")
        else:
            get_ipython().system("{BCFTOOLS} annotate --remove FORMAT/DP,FORMAT/GQ,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} norm -m -any -Oz -o {intermediate_fns['biallelic']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['biallelic']}")

    if rewrite or not os.path.exists(intermediate_fns['regenotyped']):
        get_ipython().system("/nfs/team112/software/htslib/vfp/vfp_tool {intermediate_fns['biallelic']} {vfp_tool_configs[vfp_tool_config]} |         bgzip -c > {intermediate_fns['regenotyped']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['regenotyped']}")

    if rewrite or not os.path.exists(intermediate_fns['new_af']):
        get_ipython().system("{BCFTOOLS} view --samples {all_samples} -Oz -o {intermediate_fns['new_af']} {intermediate_fns['regenotyped']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['new_af']}")

    if rewrite or not os.path.exists(intermediate_fns['nonref']):
        get_ipython().system('{BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'nonref\']} {intermediate_fns[\'new_af\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['nonref']}")

    if rewrite or not os.path.exists(intermediate_fns['pass']):
        get_ipython().system("{BCFTOOLS} view -f PASS -Oz -o {intermediate_fns['pass']} {intermediate_fns['nonref']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['pass']}")

    if rewrite or not os.path.exists(intermediate_fns['minimal']):
        get_ipython().system("{BCFTOOLS} annotate --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set, -Oz -o {intermediate_fns['minimal']} {intermediate_fns['pass']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['minimal']}")

    if rewrite or not os.path.exists(intermediate_fns['analysis']):
        get_ipython().system("{BCFTOOLS} norm --fasta-ref {GENOME_FN} -Oz -o {intermediate_fns['analysis']} {intermediate_fns['minimal']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['analysis']}")
        
    if rewrite or not os.path.exists(intermediate_fns['SNP']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp"\' -Oz -o {intermediate_fns[\'SNP\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP']}")

    if rewrite or not os.path.exists(intermediate_fns['SNP_BIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType="SNP"\' -Oz -o {intermediate_fns[\'SNP_BIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP_BIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['SNP_MULTIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType="MULTIALLELIC_SNP"\' -Oz -o {intermediate_fns[\'SNP_MULTIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP_MULTIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['SNP_MIXED']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType!="SNP" && VariantType!="MULTIALLELIC_SNP"\' -Oz -o {intermediate_fns[\'SNP_MIXED\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['SNP_MIXED']}")

    if rewrite or not os.path.exists(intermediate_fns['INDEL']):
        get_ipython().system('{BCFTOOLS} view --exclude \'TYPE="snp"\' -Oz -o {intermediate_fns[\'INDEL\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['INDEL']}")

    if rewrite or not os.path.exists(intermediate_fns['INDEL_BIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE!="snp" && VariantType!~"^MULTIALLELIC"\' -Oz -o {intermediate_fns[\'INDEL_BIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['INDEL_BIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['INDEL_MULTIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE!="snp" && VariantType~"^MULTIALLELIC"\' -Oz -o {intermediate_fns[\'INDEL_MULTIALLELIC\']} {intermediate_fns[\'analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['INDEL_MULTIALLELIC']}")
        
    if rewrite or not os.path.exists(intermediate_fns['gatk_new_af']):
        get_ipython().system("{BCFTOOLS} view --samples {all_samples} -Oz -o {intermediate_fns['gatk_new_af']} {intermediate_fns['biallelic']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_new_af']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_nonref']):
        get_ipython().system('{BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'gatk_nonref\']} {intermediate_fns[\'gatk_new_af\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_nonref']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_pass']):
        get_ipython().system("{BCFTOOLS} view -f PASS -Oz -o {intermediate_fns['gatk_pass']} {intermediate_fns['gatk_nonref']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_pass']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_minimal']):
        get_ipython().system("{BCFTOOLS} annotate --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set, -Oz -o {intermediate_fns['gatk_minimal']} {intermediate_fns['gatk_pass']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_minimal']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_analysis']):
        get_ipython().system("{BCFTOOLS} norm --fasta-ref {GENOME_FN} -Oz -o {intermediate_fns['gatk_analysis']} {intermediate_fns['gatk_minimal']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_analysis']}")
        
    if rewrite or not os.path.exists(intermediate_fns['gatk_SNP']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp"\' -Oz -o {intermediate_fns[\'gatk_SNP\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_SNP']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_SNP_BIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType="SNP"\' -Oz -o {intermediate_fns[\'gatk_SNP_BIALLELIC\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_SNP_BIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_SNP_MULTIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType="MULTIALLELIC_SNP"\' -Oz -o {intermediate_fns[\'gatk_SNP_MULTIALLELIC\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_SNP_MULTIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_SNP_MIXED']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE="snp" && VariantType!="SNP" && VariantType!="MULTIALLELIC_SNP"\' -Oz -o {intermediate_fns[\'gatk_SNP_MIXED\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_SNP_MIXED']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_INDEL']):
        get_ipython().system('{BCFTOOLS} view --exclude \'TYPE="snp"\' -Oz -o {intermediate_fns[\'gatk_INDEL\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_INDEL']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_INDEL_BIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE!="snp" && VariantType!~"^MULTIALLELIC"\' -Oz -o {intermediate_fns[\'gatk_INDEL_BIALLELIC\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_INDEL_BIALLELIC']}")

    if rewrite or not os.path.exists(intermediate_fns['gatk_INDEL_MULTIALLELIC']):
        get_ipython().system('{BCFTOOLS} view --include \'TYPE!="snp" && VariantType~"^MULTIALLELIC"\' -Oz -o {intermediate_fns[\'gatk_INDEL_MULTIALLELIC\']} {intermediate_fns[\'gatk_analysis\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['gatk_INDEL_MULTIALLELIC']}")
        
    for variant_type in ['SNP_BIALLELIC', 'SNP_MULTIALLELIC', 'SNP_MIXED', 'INDEL_BIALLELIC', 'INDEL_MULTIALLELIC',
                        'gatk_SNP_BIALLELIC', 'gatk_SNP_MULTIALLELIC', 'gatk_SNP_MIXED', 'gatk_INDEL_BIALLELIC', 'gatk_INDEL_MULTIALLELIC']:
        if rewrite or not os.path.exists("%s/.vcfnp_cache/variants.npy" % intermediate_fns[variant_type]):
            create_variants_npy(intermediate_fns[variant_type])
        if rewrite or not os.path.exists("%s/.vcfnp_cache/calldata_2d.npy" % intermediate_fns[variant_type]):
            create_calldata_npy(intermediate_fns[variant_type])
        

# create_analysis_vcf(region='Pf3D7_14_v3', rewrite=True)
create_analysis_vcf()

for region in ['Pf3D7_04_v3', 'Pf3D7_14_v3']:
    for vfp_tool_config in vfp_tool_configs:
        print(region, vfp_tool_config)
        create_analysis_vcf(region=region, vfp_tool_config=vfp_tool_config)

chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_Pf3D7_14_v3_5_2.analysis.vcf.gz" % output_dir
variants_SNP_BIALLELIC = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', 'SNP_BIALLELIC'))
calldata_SNP_BIALLELIC = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', 'SNP_BIALLELIC'))

calldata_SNP_BIALLELIC['GT'].shape[1]

calldata_SNP_BIALLELIC['GT'][:, 1] == b'0'

np.unique(variants_SNP_BIALLELIC['SNPEFF_EFFECT'], return_counts=True)

np.unique(calldata_SNP_BIALLELIC['GT'], return_counts=True)

4788/(np.sum([  1533, 196926,   4788,  95261]))

hets_per_sample = np.sum(calldata_SNP_BIALLELIC['GT'] == b'0/1', 0)
print(len(hets_per_sample))

hets_per_sample

def genotype_concordance(calldata=calldata_SNP_BIALLELIC['GT'], rep_index_first=rep_index_first,
                         rep_index_second=rep_index_second, verbose=False):
    all_samples = tbl_release5_crosses_metadata.values('sample').array()
    total_mendelian_errors = 0
    total_homozygotes = 0
    for cross in tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array():
        parents = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'parent')
            .values('sample')
            .array()
        )
        parent1_calls = calldata[:, all_samples==parents[0]]
#         print(parent1_calls.shape)
#         print(parent1_calls == b'1')
        parent2_calls = calldata[:, all_samples==parents[1]]
#         print(parent1_calls.shape)
        progeny = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'progeny')
            .values('sample')
            .array()
        )
        mendelian_errors = np.zeros(len(progeny))
        homozygotes = np.zeros(len(progeny))
        for i, ox_code in enumerate(progeny):
            progeny_calls = calldata[:, all_samples==ox_code]
#             print(ox_code, progeny_calls.shape)
            error_calls = (
                ((parent1_calls == b'0') & (parent2_calls == b'0') & (progeny_calls == b'1')) |
                ((parent1_calls == b'1') & (parent2_calls == b'1') & (progeny_calls == b'0'))
            )
            homozygous_calls = (
                ((parent1_calls == b'0') | (parent1_calls == b'1' )) &
                ((parent2_calls == b'0') | (parent2_calls == b'1' )) &
                ((progeny_calls == b'0') | (progeny_calls == b'1' ))
            )
            mendelian_errors[i] = np.sum(error_calls)
            homozygotes[i] = np.sum(homozygous_calls)
#         print(cross, mendelian_errors, homozygotes)
        total_mendelian_errors = total_mendelian_errors + np.sum(mendelian_errors)
        total_homozygotes = total_homozygotes + np.sum(homozygotes)
        
    mendelian_error_rate = total_mendelian_errors / total_homozygotes
    
    calldata_both = (np.in1d(calldata[:, rep_index_first], [b'0', b'1']) &
                     np.in1d(calldata[:, rep_index_second], [b'0', b'1'])
                    )
    calldata_both = (
        ((calldata[:, rep_index_first] == b'0') | (calldata[:, rep_index_first] == b'1')) &
        ((calldata[:, rep_index_second] == b'0') | (calldata[:, rep_index_second] == b'1'))
    )
    calldata_discordant = (
        ((calldata[:, rep_index_first] == b'0') & (calldata[:, rep_index_second] == b'1')) |
        ((calldata[:, rep_index_first] == b'1') & (calldata[:, rep_index_second] == b'0'))
    )
    missingness_per_sample = np.sum(calldata == b'.', 0)
    if verbose:
        print(missingness_per_sample)
    mean_missingness = np.sum(missingness_per_sample) / (calldata.shape[0] * calldata.shape[1])
    heterozygosity_per_sample = np.sum(calldata == b'0/1', 0)
    if verbose:
        print(heterozygosity_per_sample)
#     print(heterozygosity_per_sample)
    mean_heterozygosity = np.sum(heterozygosity_per_sample) / (calldata.shape[0] * calldata.shape[1])
#     print(calldata_both.shape)
#     print(calldata_discordant.shape)
    num_discordance_per_sample_pair = np.sum(calldata_discordant, 0)
    num_both_calls_per_sample_pair = np.sum(calldata_both, 0)
    if verbose:
        print(num_discordance_per_sample_pair)
        print(num_both_calls_per_sample_pair)
    prop_discordances_per_sample_pair = num_discordance_per_sample_pair/num_both_calls_per_sample_pair
    num_discordance_per_snp = np.sum(calldata_discordant, 1)
#     print(num_discordance_per_snp)
#     print(variants_SNP_BIALLELIC[num_discordance_per_snp > 0])
    mean_prop_discordances = (np.sum(num_discordance_per_sample_pair) / np.sum(num_both_calls_per_sample_pair))
    return(
        mean_missingness,
        mean_heterozygosity,
        mean_prop_discordances,
        mendelian_error_rate,
        prop_discordances_per_sample_pair,
        calldata.shape
    )
    

def genotype_concordance_gatk(calldata=calldata_SNP_BIALLELIC['GT'], rep_index_first=rep_index_first,
                         rep_index_second=rep_index_second, verbose=False):
    all_samples = tbl_release5_crosses_metadata.values('sample').array()
    total_mendelian_errors = 0
    total_homozygotes = 0
    for cross in tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array():
        parents = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'parent')
            .values('sample')
            .array()
        )
        parent1_calls = calldata[:, all_samples==parents[0]]
#         print(parent1_calls.shape)
#         print(parent1_calls == b'1')
        parent2_calls = calldata[:, all_samples==parents[1]]
#         print(parent1_calls.shape)
        progeny = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'progeny')
            .values('sample')
            .array()
        )
        mendelian_errors = np.zeros(len(progeny))
        homozygotes = np.zeros(len(progeny))
        for i, ox_code in enumerate(progeny):
            progeny_calls = calldata[:, all_samples==ox_code]
#             print(ox_code, progeny_calls.shape)
            error_calls = (
                ((parent1_calls == b'0/0') & (parent2_calls == b'0/0') & (progeny_calls == b'1/1')) |
                ((parent1_calls == b'1/1') & (parent2_calls == b'1/1') & (progeny_calls == b'0/0'))
            )
            homozygous_calls = (
                ((parent1_calls == b'0/0') | (parent1_calls == b'1/1' )) &
                ((parent2_calls == b'0/0') | (parent2_calls == b'1/1' )) &
                ((progeny_calls == b'0/0') | (progeny_calls == b'1/1' ))
            )
            mendelian_errors[i] = np.sum(error_calls)
            homozygotes[i] = np.sum(homozygous_calls)
#         print(cross, mendelian_errors, homozygotes)
        total_mendelian_errors = total_mendelian_errors + np.sum(mendelian_errors)
        total_homozygotes = total_homozygotes + np.sum(homozygotes)
        
    mendelian_error_rate = total_mendelian_errors / total_homozygotes
    
    calldata_both = (np.in1d(calldata[:, rep_index_first], [b'0/0', b'1/1']) &
                     np.in1d(calldata[:, rep_index_second], [b'0/0', b'1/1'])
                    )
    calldata_both = (
        ((calldata[:, rep_index_first] == b'0/0') | (calldata[:, rep_index_first] == b'1/1')) &
        ((calldata[:, rep_index_second] == b'0/0') | (calldata[:, rep_index_second] == b'1/1'))
    )
    calldata_discordant = (
        ((calldata[:, rep_index_first] == b'0/0') & (calldata[:, rep_index_second] == b'1/1')) |
        ((calldata[:, rep_index_first] == b'1/1') & (calldata[:, rep_index_second] == b'0/0'))
    )
    missingness_per_sample = np.sum(calldata == b'./.', 0)
    if verbose:
        print(missingness_per_sample)
    mean_missingness = np.sum(missingness_per_sample) / (calldata.shape[0] * calldata.shape[1])
    heterozygosity_per_sample = np.sum(calldata == b'0/1', 0)
    if verbose:
        print(heterozygosity_per_sample)
#     print(heterozygosity_per_sample)
    mean_heterozygosity = np.sum(heterozygosity_per_sample) / (calldata.shape[0] * calldata.shape[1])
#     print(calldata_both.shape)
#     print(calldata_discordant.shape)
    num_discordance_per_sample_pair = np.sum(calldata_discordant, 0)
    num_both_calls_per_sample_pair = np.sum(calldata_both, 0)
    if verbose:
        print(num_discordance_per_sample_pair)
        print(num_both_calls_per_sample_pair)
    prop_discordances_per_sample_pair = num_discordance_per_sample_pair/num_both_calls_per_sample_pair
    num_discordance_per_snp = np.sum(calldata_discordant, 1)
#     print(num_discordance_per_snp)
#     print(variants_SNP_BIALLELIC[num_discordance_per_snp > 0])
    mean_prop_discordances = (np.sum(num_discordance_per_sample_pair) / np.sum(num_both_calls_per_sample_pair))
    return(
        mean_missingness,
        mean_heterozygosity,
        mean_prop_discordances,
        mendelian_error_rate,
        prop_discordances_per_sample_pair,
        calldata.shape
    )
    

for region in ['Pf3D7_04_v3', 'Pf3D7_14_v3']:
    for vfp_tool_config in vfp_tool_configs:
        chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_%s_%s.analysis.vcf.gz" % (output_dir, region, vfp_tool_config)
        for variant_type in ['SNP_BIALLELIC', 'SNP_MULTIALLELIC', 'SNP_MIXED', 'INDEL_BIALLELIC', 'INDEL_MULTIALLELIC']:
            variants = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
            calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
        #     print(variant_type)
            mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'])
            print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)
#             mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'], quad_index_first, quad_index_second)
#             print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)

for region in ['Pf3D7_04_v3', 'Pf3D7_14_v3']:
    for vfp_tool_config in vfp_tool_configs:
        chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_%s_%s.analysis.vcf.gz" % (output_dir, region, vfp_tool_config)
        for variant_type in ['gatk_SNP_BIALLELIC', 'gatk_SNP_MULTIALLELIC', 'gatk_SNP_MIXED', 'gatk_INDEL_BIALLELIC', 'gatk_INDEL_MULTIALLELIC']:
            variants = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
            calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
        #     print(variant_type)
            mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance_gatk(calldata['GT'])
            print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)
#             mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'], quad_index_first, quad_index_second)
#             print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)

for region in ['Pf3D7_14_v3']:
    for vfp_tool_config in ['5_2']:
        chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_%s_%s.analysis.vcf.gz" % (output_dir, region, vfp_tool_config)
        for variant_type in ['SNP_BIALLELIC', 'SNP_MULTIALLELIC', 'SNP_MIXED', 'INDEL_BIALLELIC', 'INDEL_MULTIALLELIC']:
            variants = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
            calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
        #     print(variant_type)
            mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'])
            print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)
#             mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'], quad_index_first, quad_index_second)
#             print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)

for region in ['Pf3D7_14_v3']:
    for vfp_tool_config in ['5_2']:
        chrom_analysis_vcf_fn = "%s/vcf/SNP_INDEL_%s_%s.analysis.vcf.gz" % (output_dir, region, vfp_tool_config)
        for variant_type in ['gatk_SNP_BIALLELIC', 'gatk_SNP_MULTIALLELIC', 'gatk_SNP_MIXED', 'gatk_INDEL_BIALLELIC', 'gatk_INDEL_MULTIALLELIC']:
            variants = np.load("%s.vcfnp_cache/variants.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
            calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', variant_type))
        #     print(variant_type)
            mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance_gatk(calldata['GT'])
            print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)
#             mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, prop_discordances_per_sample_pair, dims = genotype_concordance(calldata['GT'], quad_index_first, quad_index_second)
#             print(region, vfp_tool_config, variant_type, mean_missingness, mean_heterozygosity, mean_discordance, mendelian_error_rate, dims)

calldata_SNP_BIALLELIC = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', 'gatk_SNP_BIALLELIC'))
np.unique(calldata_SNP_BIALLELIC['GT'], return_counts=True)
print(calldata_SNP_BIALLELIC.shape)

1136/(1899+173626+1136+98719)

chrom_analysis_vcf_fn

calldata_SNP_BIALLELIC = np.load("%s.vcfnp_cache/calldata_2d.npy" % chrom_analysis_vcf_fn.replace('analysis', 'SNP_BIALLELIC'))
np.unique(calldata_SNP_BIALLELIC['GT'], return_counts=True)
print(calldata_SNP_BIALLELIC.shape)

np.sum(np.array([  1899, 173626,   1136,  98719]))

np.sum(np.array([  1802, 184456,   3279,  96427]))

np.unique(calldata_SNP_BIALLELIC['GT'], return_counts=True)

genotype_concordance()

calldata_SNP_BIALLELIC[:, rep_index_first]

variants_crosses = np.load('/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/SNP_INDEL_WG.combined.filtered.crosses.vcf.gz.vcfnp_cache/variants.npy')

variants_crosses.dtype.names

np.unique(variants_crosses['VariantType'])

del(variants_crosses)
gc.collect()

2+2



