get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')

DATA_DIR

RELEASE_DIR = "%s/pacbio_1" % DATA_DIR
RESOURCES_DIR = '%s/resources' % RELEASE_DIR

# GENOME_FN = "/nfs/pathogen003/tdo/Pfalciparum/3D7/Reference/Oct2011/Pf3D7_v3.fasta" # Note this ref used by Thomas is different to other refs we have used, e.g. chromsomes aren't in numerical order
GENOME_FN = "/lustre/scratch109/malaria/pf3k_methods/resources/Pfalciparum.genome.fasta"
SNPEFF_DIR = "/lustre/scratch109/malaria/pf3k_methods/resources/snpEff"
REGIONS_FN = "/nfs/team112_internal/rp7/src/github/malariagen/pf-crosses/meta/regions-20130225.bed.gz"

RELEASE_METADATA_FN = "%s/pf3k_pacbio_1_sample_metadata.txt" % RELEASE_DIR
WG_VCF_FN = "%s/vcf/pf3k_pacbio_1.vcf.gz" % RELEASE_DIR

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'
PICARD = 'java -jar /nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/picard/picard-tools-1.137/picard.jar'

VRPIPE_FOFN = "%s/pf3k_pacbio_1.fofn" % RELEASE_DIR
HC_INPUT_FOFN = "%s/pf3k_pacbio_1_hc_input.fofn" % RELEASE_DIR

print(WG_VCF_FN)

chromosomes = ["Pf3D7_%02d_v3" % x for x in range(1, 15, 1)]
#     'Pf3D7_API_v3', 'Pf_M76611'
# ]
chromosome_vcfs = ["%s/vcf/SNP_INDEL_%s.combined.filtered.vcf.gz" % (RELEASE_DIR, x) for x in chromosomes]
chromosome_vcfs

if not os.path.exists(RESOURCES_DIR):
    os.makedirs(RESOURCES_DIR)

get_ipython().system('cp {GENOME_FN}* {RESOURCES_DIR}')
get_ipython().system('cp -R {SNPEFF_DIR} {RESOURCES_DIR}')
get_ipython().system('cp -R {REGIONS_FN} {RESOURCES_DIR}')

for lustre_dir in ['temp', 'input', 'output', 'meta']:
    new_dir = "/lustre/scratch109/malaria/pf3k_pacbio/%s" % lustre_dir
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

test1 = '/lustre/scratch108/parasites/tdo/Pfalciparum/PF3K/Reference12Genomes/Mapping_wholeChromosomes/ReRun_Splitting_50k_14072016/Res.Mapped.Pf7G8_01.01.bam'
test2 = '/lustre/scratch108/parasites/tdo/Pfalciparum/PF3K/Reference12Genomes/Mapping_wholeChromosomes/ReRun_Splitting_50k_14072016/Res.Mapped.Pf7G8_02.02.bam'
test3 = '/lustre/scratch108/parasites/tdo/Pfalciparum/PF3K/Reference12Genomes/Mapping_wholeChromosomes/ReRun_Splitting_50k_14072016/Res.Mapped.Pf7G8_03.03.bam'
test_dir = '/lustre/scratch111/malaria/rp7/temp'
get_ipython().system('mkdir -p {test_dir}')

get_ipython().system('samtools view -H {test1}')

get_ipython().system('{PICARD} AddOrReplaceReadGroups I={test1} O={test_dir}/Pf7G8_01.bam RGID=Pf7G8 RGSM=Pf7G8 RGLB=Pf7G8 RGPU=Pf7G8 RGPL=illumina')
get_ipython().system('{PICARD} AddOrReplaceReadGroups I={test2} O={test_dir}/Pf7G8_02.bam RGID=Pf7G8 RGSM=Pf7G8 RGLB=Pf7G8 RGPU=Pf7G8 RGPL=illumina')
get_ipython().system('{PICARD} AddOrReplaceReadGroups I={test3} O={test_dir}/Pf7G8_03.bam RGID=Pf7G8 RGSM=Pf7G8 RGLB=Pf7G8 RGPU=Pf7G8 RGPL=illumina')

get_ipython().system('{PICARD} MergeSamFiles I={test_dir}/Pf7G8_01.bam I={test_dir}/Pf7G8_02.bam I={test_dir}/Pf7G8_03.bam O={test_dir}/temp.bam MERGE_SEQUENCE_DICTIONARIES=true')

17*14

thomas_bam_sample_ids = ['Pf3D7II', 'Pf7G8', 'PfCD01', 'PfDd2', 'PfGA01', 'PfGB4', 'PfGN01', 'PfHB3',
                         'PfIT', 'PfKE01', 'PfKH01', 'PfKH02', 'PfML01', 'PfSD01', 'PfSN01', 'PfTG01']

for sample_id in thomas_bam_sample_ids:
    for chrom in range(1, 15):
        sample_chrom = '%s_%02d' % (sample_id, chrom)
        print(sample_chrom)
        input_bam_fn = '/lustre/scratch108/parasites/tdo/Pfalciparum/PF3K/Reference12Genomes/Mapping_wholeChromosomes/ReRun_Splitting_50k_14072016/Res.Mapped.%s_%02d.%02d.bam' % (
            sample_id, chrom, chrom
        )
        get_ipython().system('{PICARD} AddOrReplaceReadGroups I={input_bam_fn} O=/lustre/scratch109/malaria/pf3k_pacbio/temp/{sample_chrom}.bam RGID={sample_id} RGSM={sample_id} RGLB={sample_id} RGPU={sample_id} RGPL=illumina')

for sample_id in thomas_bam_sample_ids:
    input_bams = ['I=/lustre/scratch109/malaria/pf3k_pacbio/temp/%s_%02d.bam' % (sample_id, x) for x in range(1, 15)]
    output_bam = '/lustre/scratch109/malaria/pf3k_pacbio/input/%s.bam' % sample_id
    get_ipython().system("{PICARD} MergeSamFiles {' '.join(input_bams)} O={output_bam} MERGE_SEQUENCE_DICTIONARIES=true")
    get_ipython().system('{PICARD} BuildBamIndex I={output_bam}')

sample_id = 'PfSD01'
input_bams = ['I=/lustre/scratch109/malaria/pf3k_pacbio/temp/%s_%02d.bam' % (sample_id, x) for x in
              [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
output_bam = '/lustre/scratch109/malaria/pf3k_pacbio/input/%s.bam' % sample_id
get_ipython().system("{PICARD} MergeSamFiles {' '.join(input_bams)} O={output_bam} MERGE_SEQUENCE_DICTIONARIES=true")
get_ipython().system('{PICARD} BuildBamIndex I={output_bam}')

fo = open(VRPIPE_FOFN, 'w')
print('path\tsample', file=fo)
fo.close()
fo = open(VRPIPE_FOFN, 'a')
for sample_id in thomas_bam_sample_ids:
    output_bam = '/lustre/scratch109/malaria/pf3k_pacbio/input/%s.bam' % sample_id
    print('%s\t%s' % (output_bam, sample_id), file=fo)
fo.close()

# Removed PfSD01 as no reads for chromsome 4 - see email from Thomas 14/07/2016 20:17 and response 17/07/2016 11:37
thomas_bam_sample_ids = ['Pf3D7II', 'Pf7G8', 'PfCD01', 'PfDd2', 'PfGA01', 'PfGB4', 'PfGN01', 'PfHB3',
                         'PfIT', 'PfKE01', 'PfKH01', 'PfKH02', 'PfML01', 'PfSN01', 'PfTG01']

fo = open(HC_INPUT_FOFN, 'w')
for sample_id in thomas_bam_sample_ids:
    output_bam = '/lustre/scratch109/malaria/pf3k_pacbio/input/%s.bam' % sample_id
    print('%s' % (output_bam), file=fo)
fo.close()

VRPIPE_FOFN

HC_INPUT_FOFN

get_ipython().system("grep -P 'Pf3D7_[0|1]' /lustre/scratch109/malaria/pf3k_methods/resources/Pfalciparum.genome.fasta.fai > /lustre/scratch109/malaria/pf3k_pacbio/Pfalciparum.autosomes.fasta.fai")

get_ipython().system('{BCFTOOLS} concat {" ".join(chromosome_vcfs)} | sed \'s/##FORMAT=<ID=AD,Number=./##FORMAT=<ID=AD,Number=R/\' | bgzip -c > {WG_VCF_FN}')
get_ipython().system('tabix -p vcf {WG_VCF_FN}')

WG_VCF_FN

number_of_snps = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'TYPE="snp"\' {WG_VCF_FN} | wc -l')
number_of_indels = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'TYPE!="snp"\' {WG_VCF_FN} | wc -l')
number_of_pass_snps = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE="snp"\' {WG_VCF_FN} | wc -l')
number_of_pass_indels = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE!="snp"\' {WG_VCF_FN} | wc -l')
number_of_pass_biallelic_snps = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE="snp" && N_ALT=1\' {WG_VCF_FN} | wc -l')
number_of_pass_biallelic_indels = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE!="snp" && N_ALT=1\' {WG_VCF_FN} | wc -l')
number_of_VQSLODgt6_snps = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE="snp" && VQSLOD>6\' {WG_VCF_FN} | wc -l')
number_of_VQSLODgt6_indels = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE!="snp" && VQSLOD>6\' {WG_VCF_FN} | wc -l')

print("%s variants" % ("{:,}".format(int(number_of_snps[0]) + int(number_of_indels[0]))))
print("%s SNPs" % ("{:,}".format(int(number_of_snps[0]))))
print("%s indels" % ("{:,}".format(int(number_of_indels[0]))))
print()
print("%s PASS variants" % ("{:,}".format(int(number_of_pass_snps[0]) + int(number_of_pass_indels[0]))))
print("%s PASS SNPs" % ("{:,}".format(int(number_of_pass_snps[0]))))
print("%s PASS indels" % ("{:,}".format(int(number_of_pass_indels[0]))))
print()
print("%s PASS biallelic variants" % ("{:,}".format(int(number_of_pass_biallelic_snps[0]) + int(number_of_pass_biallelic_indels[0]))))
print("%s PASS biallelic SNPs" % ("{:,}".format(int(number_of_pass_biallelic_snps[0]))))
print("%s PASS biallelic indels" % ("{:,}".format(int(number_of_pass_biallelic_indels[0]))))
print()
print("%s VQSLOD>6.0 variants" % ("{:,}".format(int(number_of_VQSLODgt6_snps[0]) + int(number_of_VQSLODgt6_indels[0]))))
print("%s VQSLOD>6.0 SNPs" % ("{:,}".format(int(number_of_VQSLODgt6_snps[0]))))
print("%s VQSLOD>6.0 indels" % ("{:,}".format(int(number_of_VQSLODgt6_indels[0]))))
print()

"{number_of_pass_variants}/{number_of_variants} variants ({pct_pass}%) pass all filters".format(
        number_of_variants="{:,}".format(int(number_of_snps[0]) + int(number_of_indels[0])),
        number_of_pass_variants="{:,}".format(int(number_of_pass_snps[0]) + int(number_of_pass_indels[0])),
        pct_pass=round((
            (int(number_of_pass_snps[0]) + int(number_of_pass_indels[0])) /
            (int(number_of_snps[0]) + int(number_of_indels[0]))
        ) * 100)
)

print('''
The VCF file contains details of {number_of_variants} discovered variants of which {number_of_snps}
are SNPs and {number_of_indels} are indels (or multi-allelic mixtures of SNPs
and indels). It is important to note that many of these variants are
considered low quality. Only the variants for which the FILTER column is set
to PASS should be considered of high quality. There are {number_of_pass_variants} such high-
quality PASS variants ({number_of_pass_snps} SNPs and {number_of_pass_indels} indels).

The FILTER column is based on two types of information. Firstly certain regions
of the genome are considered "non-core". This includes sub-telomeric regions,
centromeres and internal VAR gene clusters on chromosomes 4, 6, 7, 8 and 12.
The apicoplast and mitochondrion are also considered non-core. All variants within
non-core regions are considered to be low quality, and hence will not have the
FILTER column set to PASS. The regions which are core and non-core can be found
in the file resources/regions-20130225.bed.gz.

Secondly, variants are filtered out based on a quality score called VQSLOD. All
variants with a VQSLOD score below 0 are filtered out, i.e. will have a value of
Low_VQSLOD in the FILTER column, rather than PASS. The VQSLOD score for each
variant can be found in the INFO field of the VCF file. It is possible to use the
VQSLOD score to define a more or less stringent set of variants. For example for
a very stringent set of the highest quality variants, select only those variants
where VQSLOD >= 6. There are {number_of_VQSLODgt6_snps} such stringent SNPs and {number_of_VQSLODgt6_indels}
such stringent indels.

It is also important to note that some variants have more than two alleles. For
example, amongst the {number_of_pass_snps} high quality PASS SNPs, {number_of_pass_biallelic_snps} are biallelic. The
remaining {number_of_pass_multiallelic_snps} high quality PASS SNPs have 3 or more alleles. Similarly, amongst
the {number_of_pass_indels} high-quality PASS indels, {number_of_pass_biallelic_indels} are biallelic. The remaining
{number_of_pass_multiallelic_indels} high quality PASS indels have 3 or more alleles.
'''.format(
        number_of_variants="{:,}".format(int(number_of_snps[0]) + int(number_of_indels[0])),
        number_of_snps="{:,}".format(int(number_of_snps[0])),
        number_of_indels="{:,}".format(int(number_of_indels[0])),
        number_of_pass_variants="{:,}".format(int(number_of_pass_snps[0]) + int(number_of_pass_indels[0])),
        number_of_pass_snps="{:,}".format(int(number_of_pass_snps[0])),
        number_of_pass_indels="{:,}".format(int(number_of_pass_indels[0])),
        number_of_VQSLODgt6_snps="{:,}".format(int(number_of_VQSLODgt6_snps[0])),
        number_of_VQSLODgt6_indels="{:,}".format(int(number_of_VQSLODgt6_indels[0])),
        number_of_pass_biallelic_snps="{:,}".format(int(number_of_pass_biallelic_snps[0])),
        number_of_pass_biallelic_indels="{:,}".format(int(number_of_pass_biallelic_indels[0])),
        number_of_pass_multiallelic_snps="{:,}".format(int(number_of_pass_snps[0]) - int(number_of_pass_biallelic_snps[0])),
        number_of_pass_multiallelic_indels="{:,}".format(int(number_of_pass_indels[0]) - int(number_of_pass_biallelic_indels[0])),
    )
)

# Seems like Jim prefers team112 to have write access, so haven't made unreadable
# !chmod -R uga-w {RELEASE_DIR}

RELEASE_DIR

