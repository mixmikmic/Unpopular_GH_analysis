get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')

release5_final_files_dir = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0'
# chrom_vcf_fn = "%s/SNP_INDEL_Pf3D7_14_v3.combined.filtered.vcf.gz" % (release5_final_files_dir)
crosses_vcf_fn = "%s/SNP_INDEL_WG.combined.filtered.crosses.vcf.gz" % (release5_final_files_dir)
sites_only_vcf_fn = "%s/SNP_INDEL_WG.combined.filtered.sites.vcf.gz" % (release5_final_files_dir)

output_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160720_mendelian_error_duplicate_concordance"
get_ipython().system('mkdir -p {output_dir}/vcf')

release5_crosses_metadata_txt_fn = '../../meta/pf3k_release_5_crosses_metadata.txt'
gff_fn = "%s/Pfalciparum.noseq.gff3.gz" % output_dir
cds_gff_fn = "%s/Pfalciparum.noseq.gff3.cds.gz" % output_dir

results_table_fn = "%s/genotype_quality.xlsx" % output_dir
counts_table_fn = "%s/variant_counts.xlsx" % output_dir

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'
GATK = '/software/jre1.7.0_25/bin/java -jar /nfs/team112_internal/production/tools/bin/gatk/GenomeAnalysisTK-3.4-46/GenomeAnalysisTK.jar'

gff_fn

get_ipython().system('wget ftp://ftp.sanger.ac.uk/pub/project/pathogens/gff3/2016-06/Pfalciparum.noseq.gff3.gz     -O {gff_fn}')

get_ipython().system("zgrep CDS {gff_fn} | sort -k1,1 -k4n,5n | cut -f 1,4,5 | sed 's/$/\\t1/' | bgzip -c > {cds_gff_fn} && tabix -s1 -b2 -e3 {cds_gff_fn}")

crosses_vcf_fn

multiallelic_header_fn = "%s/vcf/MULTIALLELIC.hdr" % (output_dir)
fo=open(multiallelic_header_fn, 'w')
print('##INFO=<ID=MULTIALLELIC,Number=1,Type=String,Description="Is position biallelic (BI), biallelic plus spanning deletion (SD) or truly multiallelic (MU)">', file=fo)
fo.close()

variant_type_header_fn = "%s/vcf/VARIANT_TYPE.hdr" % (output_dir)
fo=open(variant_type_header_fn, 'w')
print('##INFO=<ID=VARIANT_TYPE,Number=1,Type=String,Description="SNP or indel (IND)">', file=fo)
fo.close()

cds_header_fn = "%s/vcf/CDS.hdr" % (output_dir)
fo=open(cds_header_fn, 'w')
print('##INFO=<ID=CDS,Number=0,Type=Flag,Description="Is position coding">', file=fo)
fo.close()

def create_analysis_vcf(input_vcf_fn=crosses_vcf_fn, region='Pf3D7_14_v3:1000000-1100000',
                        output_vcf_fn=None, BCFTOOLS=BCFTOOLS, rewrite=False):
    if output_vcf_fn is None:
        if region is None:
            output_vcf_fn = "%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir)
        else:
            output_vcf_fn = "%s/vcf/SNP_INDEL_%s.analysis.vcf.gz" % (output_dir, region)
    intermediate_fns = collections.OrderedDict()
    for intermediate_file in ['nonref', 'multiallelic', 'triallelic', 'bi_allelic', 'spanning_deletion', 'triallelic_no_sd', 'multiallelics',
                              'biallelic', 'str', 'snps', 'indels', 'strs', 'variant_type', 'coding', 'analysis',
                             'site_snps', 'site_indels', 'site_variant_type', 'site_analysis']:
        intermediate_fns[intermediate_file] = output_vcf_fn.replace('analysis', intermediate_file)
    
    if rewrite or not os.path.exists(intermediate_fns['nonref']):
        if region is not None:
            get_ipython().system('{BCFTOOLS} annotate --regions {region} --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/STR,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'nonref\']}')
        else:
            get_ipython().system('{BCFTOOLS} annotate --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/STR,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set,FORMAT/PGT,FORMAT/PID,FORMAT/PL {input_vcf_fn} |             {BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'nonref\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['nonref']}")

    if rewrite or not os.path.exists(intermediate_fns['multiallelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\tMU\\n' --include 'N_ALT>2' {intermediate_fns['nonref']} | bgzip -c > {intermediate_fns['multiallelic']} && tabix -s1 -b2 -e2 {intermediate_fns['multiallelic']}")
        
    if rewrite or not os.path.exists(intermediate_fns['triallelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\t%ALT\\tSD\\n' --include 'N_ALT=2' {intermediate_fns['nonref']} | bgzip -c > {intermediate_fns['triallelic']} && tabix -s1 -b2 -e2 {intermediate_fns['triallelic']}")

    if rewrite or not os.path.exists(intermediate_fns['bi_allelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\t%ALT\\tBI\\n' --include 'N_ALT=1' {intermediate_fns['nonref']} | bgzip -c > {intermediate_fns['bi_allelic']} && tabix -s1 -b2 -e2 {intermediate_fns['bi_allelic']}")

    if rewrite or not os.path.exists(intermediate_fns['spanning_deletion']):
        get_ipython().system("zgrep '\\*' {intermediate_fns['triallelic']} | bgzip -c > {intermediate_fns['spanning_deletion']} && tabix -s1 -b2 -e2 {intermediate_fns['spanning_deletion']}")
        
    if rewrite or not os.path.exists(intermediate_fns['triallelic_no_sd']):
        get_ipython().system("zgrep -v '\\*' {intermediate_fns['triallelic']} | sed 's/SD/MU/g' | bgzip -c > {intermediate_fns['triallelic_no_sd']} && tabix -s1 -b2 -e2 {intermediate_fns['triallelic_no_sd']}")
        
    if rewrite or not os.path.exists(intermediate_fns['multiallelics']):
        get_ipython().system("{BCFTOOLS} annotate -a {intermediate_fns['bi_allelic']} -c CHROM,POS,-,INFO/MULTIALLELIC         -h {multiallelic_header_fn} {intermediate_fns['nonref']} |         {BCFTOOLS} annotate -a {intermediate_fns['spanning_deletion']} -c CHROM,POS,-,INFO/MULTIALLELIC |        {BCFTOOLS} annotate -a {intermediate_fns['multiallelic']} -c CHROM,POS,INFO/MULTIALLELIC |        {BCFTOOLS} annotate -a {intermediate_fns['triallelic_no_sd']} -c CHROM,POS,-,INFO/MULTIALLELIC         -Oz -o {intermediate_fns['multiallelics']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['multiallelics']}")
        
    if rewrite or not os.path.exists(intermediate_fns['biallelic']):
        get_ipython().system('{BCFTOOLS} norm -m -any --fasta-ref {GENOME_FN} {intermediate_fns[\'multiallelics\']} |         {BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {intermediate_fns[\'biallelic\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['biallelic']}")

#     if rewrite or not os.path.exists(intermediate_fns['str']):
#         !{GATK} -T VariantAnnotator \
#             -R {GENOME_FN} \
#             -o {intermediate_fns['str']} \
#             -A TandemRepeatAnnotator  \
#             -V {intermediate_fns['biallelic']}
# #         !{BCFTOOLS} index --tbi {intermediate_fns['str']}

    if rewrite or not os.path.exists(intermediate_fns['snps']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tSNP\\n\' --include \'TYPE="snp"\' {intermediate_fns[\'biallelic\']} | bgzip -c > {intermediate_fns[\'snps\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'snps\']}')

    if rewrite or not os.path.exists(intermediate_fns['indels']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tINDEL\\n\' --include \'TYPE!="snp"\' {intermediate_fns[\'biallelic\']} | bgzip -c > {intermediate_fns[\'indels\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'indels\']}')

#     if rewrite or not os.path.exists(intermediate_fns['strs']):
#         !{BCFTOOLS} query -f'%CHROM\t%POS\tSTR\n' --include 'TYPE!="snp" && STR=1' {intermediate_fns['str']} | bgzip -c > {intermediate_fns['strs']} && tabix -s1 -b2 -e2 -f {intermediate_fns['strs']}

    if rewrite or not os.path.exists(intermediate_fns['variant_type']):
        get_ipython().system("{BCFTOOLS} annotate -a {intermediate_fns['snps']} -c CHROM,POS,INFO/VARIANT_TYPE         -h {variant_type_header_fn} {intermediate_fns['biallelic']} |        {BCFTOOLS} annotate -a {intermediate_fns['indels']} -c CHROM,POS,INFO/VARIANT_TYPE         -Oz -o {intermediate_fns['variant_type']} ")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['variant_type']}")

        #         {BCFTOOLS} annotate -a {intermediate_fns['strs']} -c CHROM,POS,INFO/VARIANT_TYPE \

    if rewrite or not os.path.exists(intermediate_fns['analysis']):
        get_ipython().system("{BCFTOOLS} annotate -a {cds_gff_fn} -c CHROM,FROM,TO,CDS         -h {cds_header_fn}         -Oz -o {intermediate_fns['analysis']} {intermediate_fns['variant_type']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['analysis']}")

#     if rewrite or not os.path.exists(intermediate_fns['site_snps']):
#         !{BCFTOOLS} query -f'%CHROM\t%POS\tSNP\n' --include 'TYPE="snp"' {intermediate_fns['multiallelics']} | bgzip -c > {intermediate_fns['site_snps']} && tabix -s1 -b2 -e2 -f {intermediate_fns['site_snps']}

#     if rewrite or not os.path.exists(intermediate_fns['site_indels']):
#         !{BCFTOOLS} query -f'%CHROM\t%POS\tINDEL\n' --include 'TYPE!="snp"' {intermediate_fns['multiallelics']} | bgzip -c > {intermediate_fns['site_indels']} && tabix -s1 -b2 -e2 -f {intermediate_fns['site_indels']}

#     if rewrite or not os.path.exists(intermediate_fns['site_variant_type']):
#         !{BCFTOOLS} annotate -a {intermediate_fns['site_snps']} -c CHROM,POS,INFO/VARIANT_TYPE \
#         -h {variant_type_header_fn} {intermediate_fns['multiallelics']} | \
#         {BCFTOOLS} annotate -a {intermediate_fns['site_indels']} -c CHROM,POS,INFO/VARIANT_TYPE \
#         -Oz -o {intermediate_fns['site_variant_type']} 
#         !{BCFTOOLS} index --tbi {intermediate_fns['site_variant_type']}

#     if rewrite or not os.path.exists(intermediate_fns['site_analysis']):
#         !{BCFTOOLS} annotate -a {cds_gff_fn} -c CHROM,FROM,TO,CDS \
#         -h {cds_header_fn} \
#         -Oz -o {intermediate_fns['site_analysis']} {intermediate_fns['site_variant_type']}
#         !{BCFTOOLS} index --tbi {intermediate_fns['site_analysis']}

# create_analysis_vcf()

create_analysis_vcf(region=None)

output_dir

tbl_release5_crosses_metadata = etl.fromtsv(release5_crosses_metadata_txt_fn)
print(len(tbl_release5_crosses_metadata.data()))
tbl_release5_crosses_metadata

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

# Note the version created in this notebook doesn't work. I think this is because of R in Number of FORMAT field for AD,
# which is part of spec for v4.2, but think GATK must have got rid of this in previous notebook
analysis_vcf_fn = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160719_mendelian_error_duplicate_concordance/vcf/SNP_INDEL_WG.analysis.vcf.gz"
vcf_reader = vcf.Reader(filename=analysis_vcf_fn)
sample_ids = np.array(vcf_reader.samples)

# sample_ids = tbl_release5_crosses_metadata.values('sample').array()

rep_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_first)]
rep_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], replicates_second)]
print(len(rep_index_first))
print(rep_index_first)
print(len(rep_index_second))
print(rep_index_second)

# sample_ids = tbl_release5_crosses_metadata.values('sample').array()

quad_index_first = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_first)]
quad_index_second = np.argsort(sample_ids)[np.searchsorted(sample_ids[np.argsort(sample_ids)], quads_second)]
print(len(quad_index_first))
print(quad_index_first)
print(len(quad_index_second))
print(quad_index_second)

tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array()

def create_variants_npy(vcf_fn):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.variants(
        vcf_fn,
#         fields=['CHROM', 'POS', 'REF', 'ALT', 'VariantType', 'VARIANT_TYPE', 'RU',
#                 'SNPEFF_EFFECT', 'AC', 'AN', 'RPA', 'CDS', 'MULTIALLELIC',
#                 'VQSLOD', 'FILTER'],
        fields=['CHROM', 'POS', 'REF', 'ALT', 'VariantType', 'VARIANT_TYPE',
                'SNPEFF_EFFECT', 'AC', 'AN', 'CDS', 'MULTIALLELIC',
                'VQSLOD', 'FILTER'],
        dtypes={
            'REF':                      'a10',
            'ALT':                      'a10',
            'RegionType':               'a25',
            'VariantType':              'a40',
            'VARIANT_TYPE':             'a3',
            'RU':                       'a40',
            'SNPEFF_EFFECT':            'a33',
            'CDS':                      bool,
            'MULTIALLELIC':             'a2',
        },
        arities={
            'ALT':   1,
            'AF':    1,
            'AC':    1,
            'RPA':   2,
            'ANN':   1,
        },
        fills={
            'VQSLOD': np.nan,
        },
        flatten_filter=True,
        progress=100000,
        verbose=True,
        cache=True,
        cachedir=output_dir
    )

def create_calldata_npy(vcf_fn, max_alleles=2):
    output_dir = '%s.vcfnp_cache' % vcf_fn
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vcfnp.calldata_2d(
        vcf_fn,
        fields=['GT', 'AD', 'DP', 'GQ'],
        dtypes={
            'AD': 'u2',
        },
        arities={
            'AD': max_alleles,
        },
        progress=100000,
        verbose=True,
        cache=True,
        cachedir=output_dir
    )

# create_analysis_vcf(region='Pf3D7_14_v3', rewrite=True)
# create_variants_npy("%s/vcf/SNP_INDEL_Pf3D7_14_v3:1000000-1100000.coding.vcf.gz" % (output_dir))
# create_calldata_npy("%s/vcf/SNP_INDEL_Pf3D7_14_v3:1000000-1100000.coding.vcf.gz" % (output_dir))
create_variants_npy("%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir))
create_calldata_npy("%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir))

# def create_variants_npy_2(vcf_fn):
#     output_dir = '%s.vcfnp_cache' % vcf_fn
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     vcfnp.variants(
#         vcf_fn,
#         fields=['CHROM', 'POS', 'REF', 'ALT', 'VariantType', 'VARIANT_TYPE',
#                 'SNPEFF_EFFECT', 'AC', 'AN', 'CDS', 'MULTIALLELIC',
#                 'VQSLOD', 'FILTER'],
#         dtypes={
#             'REF':                      'a10',
#             'ALT':                      'a10',
#             'RegionType':               'a25',
#             'VariantType':              'a40',
#             'VARIANT_TYPE':             'a3',
#             'SNPEFF_EFFECT':            'a33',
#             'CDS':                      bool,
#             'MULTIALLELIC':             'a2',
#         },
#         arities={
#             'ALT':   1,
#             'AF':    1,
#             'AC':    1,
#             'ANN':   1,
#         },
#         fills={
#             'VQSLOD': np.nan,
#         },
#         flatten_filter=True,
#         progress=100000,
#         verbose=True,
#         cache=True,
#         cachedir=output_dir
#     )

# create_variants_npy_2("%s/vcf/SNP_INDEL_WG.site_analysis.vcf.gz" % (output_dir))
# create_calldata_npy("%s/vcf/SNP_INDEL_WG.site_analysis.vcf.gz" % (output_dir))

analysis_vcf_fn = "%s/vcf/SNP_INDEL_WG.analysis.vcf.gz" % (output_dir)
variants = np.load("%s.vcfnp_cache/variants.npy" % analysis_vcf_fn)
calldata = np.load("%s.vcfnp_cache/calldata_2d.npy" % analysis_vcf_fn)

print(np.unique(variants['VARIANT_TYPE'], return_counts=True))

# site_analysis_vcf_fn = "%s/vcf/SNP_INDEL_WG.site_analysis.vcf.gz" % (output_dir)
# site_variants = np.load("%s.vcfnp_cache/variants.npy" % site_analysis_vcf_fn)

# print(np.unique(site_variants['VARIANT_TYPE'], return_counts=True))
# print(np.unique(site_variants['MULTIALLELIC'], return_counts=True))
# print(np.unique(site_variants['CDS'], return_counts=True))
# print(np.unique(site_variants['FILTER_PASS'], return_counts=True))

np.unique(variants['MULTIALLELIC'], return_counts=True)

def genotype_concordance_gatk(calldata=calldata,
                              ix = ((variants['VARIANT_TYPE'] == b'SNP') & (variants['MULTIALLELIC'] == b'BI') &
                                    (variants['CDS']) & variants['FILTER_PASS']),
                              GQ_threshold=30,
                              rep_index_first=rep_index_first, rep_index_second=rep_index_second,
                              verbose=False):
    GT = calldata['GT'][ix, :]
    GT[calldata['GQ'][ix, :] < GQ_threshold] = b'./.'
    
    all_samples = sample_ids
#     all_samples = tbl_release5_crosses_metadata.values('sample').array()
    total_mendelian_errors = 0
    total_homozygotes = 0
    for cross in tbl_release5_crosses_metadata.distinct('study_title').values('study_title').array():
        parents = (tbl_release5_crosses_metadata
            .selecteq('study_title', cross)
            .selecteq('parent_or_progeny', 'parent')
            .values('sample')
            .array()
        )
        parent1_calls = GT[:, all_samples==parents[0]]
#         print(parent1_calls.shape)
#         print(parent1_calls == b'1')
        parent2_calls = GT[:, all_samples==parents[1]]
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
            progeny_calls = GT[:, all_samples==ox_code]
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
    
    GT_both = (np.in1d(GT[:, rep_index_first], [b'0/0', b'1/1']) &
                     np.in1d(GT[:, rep_index_second], [b'0/0', b'1/1'])
                    )
    GT_both = (
        ((GT[:, rep_index_first] == b'0/0') | (GT[:, rep_index_first] == b'1/1')) &
        ((GT[:, rep_index_second] == b'0/0') | (GT[:, rep_index_second] == b'1/1'))
    )
    GT_discordant = (
        ((GT[:, rep_index_first] == b'0/0') & (GT[:, rep_index_second] == b'1/1')) |
        ((GT[:, rep_index_first] == b'1/1') & (GT[:, rep_index_second] == b'0/0'))
    )
    missingness_per_sample = np.sum(GT == b'./.', 0)
    if verbose:
        print(missingness_per_sample)
    mean_missingness = np.sum(missingness_per_sample) / (GT.shape[0] * GT.shape[1])
    heterozygosity_per_sample = np.sum(GT == b'0/1', 0)
    if verbose:
        print(heterozygosity_per_sample)
#     print(heterozygosity_per_sample)
    mean_heterozygosity = np.sum(heterozygosity_per_sample) / (GT.shape[0] * GT.shape[1])
#     print(calldata_both.shape)
#     print(calldata_discordant.shape)
    num_discordance_per_sample_pair = np.sum(GT_discordant, 0)
    num_both_calls_per_sample_pair = np.sum(GT_both, 0)
    if verbose:
        print(num_discordance_per_sample_pair)
        print(num_both_calls_per_sample_pair)
    prop_discordances_per_sample_pair = num_discordance_per_sample_pair/num_both_calls_per_sample_pair
    num_discordance_per_snp = np.sum(GT_discordant, 1)
#     print(num_discordance_per_snp)
#     print(variants_SNP_BIALLELIC[num_discordance_per_snp > 0])
    mean_prop_discordances = (np.sum(num_discordance_per_sample_pair) / np.sum(num_both_calls_per_sample_pair))
    num_of_alleles = np.sum(ix)
    return(
#         num_of_alleles,
        mean_missingness,
        mean_heterozygosity,
        mean_prop_discordances,
        mendelian_error_rate,
#         prop_discordances_per_sample_pair,
#         GT.shape
    )
    

genotype_concordance_gatk()

genotype_concordance_gatk()

genotype_concordance_gatk()

genotype_concordance_gatk()

results_list = list()
# GQ_thresholds = [30, 99, 0]
GQ_thresholds = [0, 30, 99]
variant_types = [b'SNP', b'IND']
# variant_types = [b'SNP', b'IND', b'STR']
multiallelics = [b'BI', b'SD', b'MU']
codings = [True, False]
filter_passes = [True, False]

for GQ_threshold in GQ_thresholds:
    for filter_pass in filter_passes:
        for variant_type in variant_types:
            for coding in codings:
                for multiallelic in multiallelics:
                    print(GQ_threshold, filter_pass, variant_type, coding, multiallelic)
                    ix = (
                        (variants['VARIANT_TYPE'] == variant_type) &
                        (variants['MULTIALLELIC'] == multiallelic) &
                        (variants['CDS'] == coding) &
                        (variants['FILTER_PASS'] == filter_pass)
                    )
                    number_of_alleles = np.sum(ix)
                    number_of_sites = len(np.unique(variants[['CHROM', 'POS']][ix]))
                    mean_nraf = np.sum(variants['AC'][ix]) / np.sum(variants['AN'][ix])
                    genotype_quality_results = list(genotype_concordance_gatk(ix=ix, GQ_threshold=GQ_threshold))
#                     sites_ix = (
#                         (site_variants['VARIANT_TYPE'] == variant_type) &
#                         (site_variants['MULTIALLELIC'] == multiallelic) &
#                         (site_variants['CDS'] == coding) &
#                         (site_variants['FILTER_PASS'] == filter_pass)
#                     )
#                     num_sites = np.sum(sites_ix)
                    results_list.append(
                        [GQ_threshold, filter_pass, variant_type, coding, multiallelic, number_of_sites, number_of_alleles, mean_nraf] +
                        genotype_quality_results
                    )

# print(results_list)

# Sanity check. Previously this was showing 2 variants, which was due to a bug
variant_type = b'SNP'
multiallelic = b'BI'
coding = False
filter_pass = True
ix = (
                        (variants['VARIANT_TYPE'] == variant_type) &
                        (variants['MULTIALLELIC'] == multiallelic) &
                        (variants['CDS'] == coding) &
                        (variants['FILTER_PASS'] == filter_pass)
                    )
temp = variants[['CHROM', 'POS']][ix]
s = np.sort(temp, axis=None)
s[s[1:] == s[:-1]]

# Sanity check. Previously this was 1 of the two variants shown above and multiallelic was b'BI' not b'MU'
variants[(variants['CHROM']==b'Pf3D7_01_v3') & (variants['POS']==514753)]

headers = ['GQ threshold', 'PASS', 'Type', 'Coding', 'Multiallelic', 'Variants', 'Alleles', 'Mean NRAF', 'Missingness',
           'Heterozygosity', 'Discordance', 'MER']
etl.wrap(results_list).pushheader(headers).displayall()

np.sum(etl.wrap(results_list).pushheader(headers).values('Variants').array())



# etl.wrap(results_list).pushheader(headers).convert('Alleles', int).toxlsx(results_table_fn)
etl.wrap(results_list).pushheader(headers).cutout('Alleles').cutout('Mean NRAF').toxlsx(results_table_fn)
results_table_fn





# variant_type_header_2_fn = "%s/vcf/VARIANT_TYPE_2.hdr" % (output_dir)
# fo=open(variant_type_header_2_fn, 'w')
# print('##INFO=<ID=VARIANT_TYPE,Number=1,Type=String,Description="SNP or INDEL">', file=fo)
# fo.close()

def create_variant_counts_vcf(input_vcf_fn=sites_only_vcf_fn, region='Pf3D7_14_v3:1000000-1100000',
                        output_vcf_fn=None, BCFTOOLS=BCFTOOLS, rewrite=False):
    if output_vcf_fn is None:
        if region is None:
            output_vcf_fn = "%s/vcf/SNP_INDEL_WG.sites.analysis.vcf.gz" % (output_dir)
        else:
            output_vcf_fn = "%s/vcf/SNP_INDEL_%s.sites.analysis.vcf.gz" % (output_dir, region)
    intermediate_fns = collections.OrderedDict()
    for intermediate_file in ['multiallelic', 'multiallelics', 'snps', 'indels', 'triallelic', 'bi_allelic', 'spanning_deletion',
                              'triallelic_no_sd', 'biallelic',
                              'variant_type', 'analysis']:
        intermediate_fns[intermediate_file] = output_vcf_fn.replace('analysis', intermediate_file)

    if rewrite or not os.path.exists(intermediate_fns['multiallelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\tMU\\n' --include 'N_ALT>2' {input_vcf_fn} | bgzip -c > {intermediate_fns['multiallelic']} && tabix -s1 -b2 -e2 {intermediate_fns['multiallelic']}")
        
    if rewrite or not os.path.exists(intermediate_fns['triallelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\t%ALT\\tSD\\n' --include 'N_ALT=2' {input_vcf_fn} | bgzip -c > {intermediate_fns['triallelic']} && tabix -s1 -b2 -e2 {intermediate_fns['triallelic']}")

    if rewrite or not os.path.exists(intermediate_fns['bi_allelic']):
        get_ipython().system("{BCFTOOLS} query -f'%CHROM\\t%POS\\t%ALT\\tBI\\n' --include 'N_ALT=1' {input_vcf_fn} | bgzip -c > {intermediate_fns['bi_allelic']} && tabix -s1 -b2 -e2 {intermediate_fns['bi_allelic']}")

    if rewrite or not os.path.exists(intermediate_fns['spanning_deletion']):
        get_ipython().system("zgrep '\\*' {intermediate_fns['triallelic']} | bgzip -c > {intermediate_fns['spanning_deletion']} && tabix -s1 -b2 -e2 {intermediate_fns['spanning_deletion']}")
        
    if rewrite or not os.path.exists(intermediate_fns['triallelic_no_sd']):
        get_ipython().system("zgrep -v '\\*' {intermediate_fns['triallelic']} | sed 's/SD/MU/g' | bgzip -c > {intermediate_fns['triallelic_no_sd']} && tabix -s1 -b2 -e2 {intermediate_fns['triallelic_no_sd']}")
        
    if rewrite or not os.path.exists(intermediate_fns['multiallelics']):
        get_ipython().system("{BCFTOOLS} annotate --remove INFO/AF,INFO/BaseQRankSum,INFO/ClippingRankSum,INFO/DP,INFO/DS,INFO/END,INFO/FS,INFO/GC,INFO/HaplotypeScore,INFO/InbreedingCoeff,INFO/MLEAC,INFO/MLEAF,INFO/MQ,INFO/MQRankSum,INFO/NEGATIVE_TRAIN_SITE,INFO/POSITIVE_TRAIN_SITE,INFO/QD,INFO/ReadPosRankSum,INFO/RegionType,INFO/RPA,INFO/RU,INFO/STR,INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EXON_ID,INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_BIOTYPE,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,INFO/SOR,INFO/culprit,INFO/set        -a {intermediate_fns['bi_allelic']} -c CHROM,POS,-,INFO/MULTIALLELIC         -h {multiallelic_header_fn} {input_vcf_fn} |         {BCFTOOLS} annotate -a {intermediate_fns['spanning_deletion']} -c CHROM,POS,-,INFO/MULTIALLELIC |        {BCFTOOLS} annotate -a {intermediate_fns['multiallelic']} -c CHROM,POS,INFO/MULTIALLELIC |        {BCFTOOLS} annotate -a {intermediate_fns['triallelic_no_sd']} -c CHROM,POS,-,INFO/MULTIALLELIC         -Oz -o {intermediate_fns['multiallelics']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['multiallelics']}")
                
#     if rewrite or not os.path.exists(intermediate_fns['multiallelics']):
#         !{BCFTOOLS} annotate -a {intermediate_fns['bi_allelic']} -c CHROM,POS,-,INFO/MULTIALLELIC \
#         -h {multiallelic_header_fn} {intermediate_fns['nonref']} | \
#         {BCFTOOLS} annotate -a {intermediate_fns['spanning_deletion']} -c CHROM,POS,-,INFO/MULTIALLELIC |\
#         {BCFTOOLS} annotate -a {intermediate_fns['multiallelic']} -c CHROM,POS,INFO/MULTIALLELIC |\
#         {BCFTOOLS} annotate -a {intermediate_fns['triallelic_no_sd']} -c CHROM,POS,-,INFO/MULTIALLELIC \
#         -Oz -o {intermediate_fns['multiallelics']}
#         !{BCFTOOLS} index --tbi {intermediate_fns['multiallelics']}
        
    if rewrite or not os.path.exists(intermediate_fns['biallelic']):
        get_ipython().system('{BCFTOOLS} norm -m -any --fasta-ref {GENOME_FN} {intermediate_fns[\'multiallelics\']} |         {BCFTOOLS} view --include \'ALT!="*"\' -Oz -o {intermediate_fns[\'biallelic\']}')
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['biallelic']}")

#         {BCFTOOLS} view --include 'AC>0 && ALT!="*"' -Oz -o {intermediate_fns['biallelic']}
        
    if rewrite or not os.path.exists(intermediate_fns['snps']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tSNP\\n\' --include \'TYPE="snp"\' {intermediate_fns[\'biallelic\']} | bgzip -c > {intermediate_fns[\'snps\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'snps\']}')

    if rewrite or not os.path.exists(intermediate_fns['indels']):
        get_ipython().system('{BCFTOOLS} query -f\'%CHROM\\t%POS\\tINDEL\\n\' --include \'TYPE!="snp"\' {intermediate_fns[\'biallelic\']} | bgzip -c > {intermediate_fns[\'indels\']} && tabix -s1 -b2 -e2 -f {intermediate_fns[\'indels\']}')

    if rewrite or not os.path.exists(intermediate_fns['variant_type']):
        get_ipython().system("{BCFTOOLS} annotate -a {intermediate_fns['snps']} -c CHROM,POS,INFO/VARIANT_TYPE         -h {variant_type_header_fn} {intermediate_fns['biallelic']} |         {BCFTOOLS} annotate -a {intermediate_fns['indels']} -c CHROM,POS,INFO/VARIANT_TYPE         -Oz -o {intermediate_fns['variant_type']} ")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['variant_type']}")

    if rewrite or not os.path.exists(intermediate_fns['analysis']):
        get_ipython().system("{BCFTOOLS} annotate -a {cds_gff_fn} -c CHROM,FROM,TO,CDS         -h {cds_header_fn}         -Oz -o {intermediate_fns['analysis']} {intermediate_fns['variant_type']}")
        get_ipython().system("{BCFTOOLS} index --tbi {intermediate_fns['analysis']}")

sites_only_vcf_fn

# create_variant_counts_vcf()

create_variant_counts_vcf(region=None)

# def create_variants_npy_2(vcf_fn):
#     output_dir = '%s.vcfnp_cache' % vcf_fn
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     vcfnp.variants(
#         vcf_fn,
#         fields=['CHROM', 'POS', 'REF', 'ALT', 'VariantType', 'VARIANT_TYPE',
#                 'SNPEFF_EFFECT', 'AC', 'AN', 'CDS', 'MULTIALLELIC',
#                 'VQSLOD', 'FILTER'],
#         dtypes={
#             'REF':                      'a10',
#             'ALT':                      'a10',
#             'RegionType':               'a25',
#             'VariantType':              'a40',
#             'VARIANT_TYPE':             'a3',
#             'SNPEFF_EFFECT':            'a33',
#             'CDS':                      bool,
#             'MULTIALLELIC':             bool,
#         },
#         arities={
#             'ALT':   1,
#             'AF':    1,
#             'AC':    1,
#             'ANN':   1,
#         },
#         fills={
#             'VQSLOD': np.nan,
#         },
#         flatten_filter=True,
#         progress=100000,
#         verbose=True,
#         cache=True,
#         cachedir=output_dir
#     )

create_variants_npy("%s/vcf/SNP_INDEL_WG.sites.analysis.vcf.gz" % (output_dir))

sites_vcf_fn = "%s/vcf/SNP_INDEL_WG.sites.analysis.vcf.gz" % (output_dir)
variants_all = np.load("%s.vcfnp_cache/variants.npy" % sites_vcf_fn)

counts_list = list()
GQ_thresholds = [0, 30, 99]
variant_types = [b'SNP', b'IND']
# variant_types = [b'SNP', b'IND', b'STR']
multiallelics = [b'BI', b'SD', b'MU']
codings = [True, False]
filter_passes = [True, False]

# GQ_thresholds = [30, 99, 0]
# variant_types = [b'SNP', b'IND']
# multiallelics = [False, True]
# codings = [True, False]
# filter_passes = [True, False]


for filter_pass in filter_passes:
    for variant_type in variant_types:
        for coding in codings:
            for multiallelic in multiallelics:
                print(filter_pass, variant_type, coding, multiallelic)
                ix = (
                    (variants_all['VARIANT_TYPE'] == variant_type) &
                    (variants_all['MULTIALLELIC'] == multiallelic) &
                    (variants_all['CDS'] == coding) &
                    (variants_all['FILTER_PASS'] == filter_pass)
                )
                number_of_alleles = np.sum(ix)
                number_of_sites = len(np.unique(variants_all[['CHROM', 'POS']][ix]))
                mean_nraf = np.sum(variants_all['AC'][ix]) / np.sum(variants_all['AN'][ix])
#                 number_of_variants = np.sum(ix)
                counts_list.append(
                    [filter_pass, variant_type, coding, multiallelic, number_of_sites, number_of_alleles, mean_nraf]
                )

print(counts_list)

headers = ['PASS', 'Type', 'Coding', 'Multiallelic', 'Variants', 'Alleles', 'Mean NRAF']
(etl
 .wrap(counts_list)
 .pushheader(headers)
 .convert('Mean NRAF', float)
 .convert('Variants', int)
 .convert('Alleles', int)
 .displayall()
)

(etl
 .wrap(counts_list)
 .pushheader(headers)
 .convert('Mean NRAF', float)
 .convert('Variants', int)
 .convert('Alleles', int)
 .toxlsx(counts_table_fn)
)
# etl.wrap(counts_list).pushheader(headers).convertnumbers().toxlsx(counts_table_fn)
counts_table_fn

np.sum(etl
       .fromtsv('/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/passcounts')
       .pushheader(('CHROM', 'count'))
       .convertnumbers()
       .values('count')
       .array()
       )

2+2



