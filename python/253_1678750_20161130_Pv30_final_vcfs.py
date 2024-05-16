get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')

output_dir = '/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pv3.0/20161130_Pv30_final_vcfs'
vrpipe_vcfs_dir = '/nfs/team112_internal/production_files/Pv/3_0/vcf'

nfs_release_dir = '/nfs/team112_internal/production/release_build/Pv/3_0_release_packages'
nfs_final_vcf_dir = '%s/vcf' % nfs_release_dir
get_ipython().system('mkdir -p {nfs_final_vcf_dir}')

gff_fn = "/lustre/scratch116/malaria/pvivax/resources/snpEff/data/PvivaxP01_GeneDB_Oct2016/PvivaxP01.noseq.gff3"
cds_gff_fn = "%s/gff/PvivaxP01_GeneDB_Oct2016.PvivaxP01.noseq.gff3.cds.gz" % output_dir
annotations_header_fn = "%s/intermediate_files/annotations.hdr" % (output_dir)

run_create_multiallelics_file_job_fn = "%s/scripts/run_create_multiallelics_file_job.sh" % output_dir
submit_create_multiallelics_file_jobs_fn = "%s/scripts/submit_create_multiallelics_file_jobs.sh" % output_dir
create_study_vcf_job_fn = "%s/scripts/create_study_vcf_job.sh" % output_dir

vrpipe_metadata_fn = "%s/Pv_3.0_vrpipe_bam_summaries.txt" % output_dir
short_contigs_fofn = "%s/short_contigs.fofn" % output_dir

GENOME_FN = "/lustre/scratch109/malaria/pvivax/resources/gatk/PvivaxP01.genome.fasta"
# BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'

genome_fn = "%s/PvivaxP01.genome.fasta" % output_dir

get_ipython().system('mkdir -p {output_dir}/gff')
get_ipython().system('mkdir -p {output_dir}/vcf')
get_ipython().system('mkdir -p {output_dir}/study_vcfs')
get_ipython().system('mkdir -p {output_dir}/intermediate_files')
get_ipython().system('mkdir -p {output_dir}/tables')
get_ipython().system('mkdir -p {output_dir}/scripts')
get_ipython().system('mkdir -p {output_dir}/log')

cds_gff_fn

get_ipython().system("grep CDS {gff_fn} | sort -k1,1 -k4n,5n | cut -f 1,4,5 | sed 's/$/\\t1/' | bgzip -c > {cds_gff_fn} && tabix -s1 -b2 -e3 {cds_gff_fn}")

fo=open(annotations_header_fn, 'w')
print('##INFO=<ID=CDS,Number=0,Type=Flag,Description="Is position coding">', file=fo)
print('##INFO=<ID=VARIANT_TYPE,Number=1,Type=String,Description="SNP or indel (INDEL)">', file=fo)
print('##INFO=<ID=MULTIALLELIC,Number=1,Type=String,Description="Is position biallelic (BI), biallelic plus spanning deletion (SD) or truly multiallelic (MU)">', file=fo)
fo.close()

fo = open(run_create_multiallelics_file_job_fn, 'w')
print('''FASTA_FAI_FILE=%s.fai
 
JOB=$LSB_JOBINDEX
# JOB=19
 
IN=`sed "$JOB q;d" $FASTA_FAI_FILE`
read -a LINE <<< "$IN"
CHROM=${LINE[0]}

INPUT_FULL_VCF_FN=%s/SNP_INDEL_$CHROM.combined.filtered.vcf.gz
INPUT_SITES_VCF_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.vcf.gz
MULTIALLELIC_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.multiallelic.txt
SNPS_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.snps.txt.gz
INDELS_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.indels.txt.gz
ANNOTATION_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.annotated.txt.gz
NORMALISED_ANNOTATION_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.annotated.normalised.vcf.gz
OUTPUT_VCF_FN=%s/vcf/Pv_30_$CHROM.final.vcf.gz

# echo $INPUT_VCF_FN
# echo $OUTPUT_TXT_FN

bcftools view --drop-genotypes -Oz -o $INPUT_SITES_VCF_FN $INPUT_FULL_VCF_FN

python /nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\ 6.0/scripts/create_multiallelics_file.py \
-i $INPUT_SITES_VCF_FN -o $MULTIALLELIC_FN

bgzip -f $MULTIALLELIC_FN && tabix -s1 -b2 -e2 $MULTIALLELIC_FN.gz

bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\tSNP\n' --include 'TYPE="snp"' $INPUT_SITES_VCF_FN | \
bgzip -c > $SNPS_FN && tabix -s1 -b2 -e2 -f $SNPS_FN

bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\tINDEL\n' --include 'TYPE!="snp"' $INPUT_SITES_VCF_FN | \
bgzip -c > $INDELS_FN && tabix -s1 -b2 -e2 -f $INDELS_FN

bcftools annotate \
-a %s -c CHROM,FROM,TO,CDS -h %s $INPUT_SITES_VCF_FN | \
bcftools annotate \
-a $SNPS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
bcftools annotate \
-a $INDELS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
bcftools annotate \
-a $MULTIALLELIC_FN.gz -c CHROM,POS,REF,ALT,INFO/MULTIALLELIC | \
bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\t%%CDS\t%%VARIANT_TYPE\t%%MULTIALLELIC\n' | \
bgzip -c > $ANNOTATION_FN

tabix -s1 -b2 -e2 $ANNOTATION_FN

#/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools index --tbi \
#$ANNOTATION_FN

bcftools norm \
-m -any --fasta-ref %s $INPUT_SITES_VCF_FN | \
bcftools view \
--include 'ALT!="*"' | \
bcftools annotate \
-h %s \
-a $ANNOTATION_FN -c CHROM,POS,REF,ALT,CDS,VARIANT_TYPE,MULTIALLELIC \
--include 'INFO/AC>0' \
--remove ^INFO/AC,INFO/AN,INFO/AF,INFO/VQSLOD -Oz -o $NORMALISED_ANNOTATION_FN

bcftools index --tbi \
$NORMALISED_ANNOTATION_FN

bcftools annotate \
-a %s -c CHROM,FROM,TO,CDS -h %s $INPUT_FULL_VCF_FN | \
bcftools annotate \
-a $SNPS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
bcftools annotate \
-a $INDELS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
bcftools annotate \
-a $MULTIALLELIC_FN.gz -c CHROM,POS,REF,ALT,INFO/MULTIALLELIC \
--remove ^INFO/AC,INFO/AF,INFO/AN,INFO/QD,INFO/MQ,INFO/FS,INFO/SOR,INFO/DP,INFO/VariantType,INFO/VQSLOD,INFO/RegionType,\
INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EFFECT,INFO/SNPEFF_EXON_ID,\
INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,\
INFO/CDS,INFO/VARIANT_TYPE,INFO/MULTIALLELIC,^FORMAT/GT,FORMAT/AD,FORMAT/DP,FORMAT/GQ,FORMAT/PGT,FORMAT/PID,FORMAT/PL,\
^FILTER/PASS,FILTER/Centromere,FILTER/InternalHypervariable,FILTER/SubtelomericHypervariable,\
FILTER/SubtelomericRepeat,FILTER/ShortContig,FILTER/Low_VQSLOD \
-Oz -o $OUTPUT_VCF_FN

bcftools index --tbi \
$OUTPUT_VCF_FN

''' % (
        GENOME_FN,
        vrpipe_vcfs_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        output_dir,
        cds_gff_fn,
        annotations_header_fn,
        GENOME_FN,
        annotations_header_fn,
        cds_gff_fn,
        annotations_header_fn,
        )
        , file=fo)
fo.close()

get_ipython().system('bash {run_create_multiallelics_file_job_fn}')

fo = open(submit_create_multiallelics_file_jobs_fn, 'w')
print('''FASTA_FAI_FILE=%s.fai
LOG_DIR=%s/log
 
NUM_CHROMS=`wc -l < $FASTA_FAI_FILE`
QUEUE=long

bsub -q $QUEUE -G malaria-dk -J "ma[1-$NUM_CHROMS]" -n2 -R"select[mem>2000] rusage[mem=2000] span[hosts=1]" -M 2000 -o $LOG_DIR/output_%%J-%%I.log %s
''' % (
        GENOME_FN,
        output_dir,
        "bash %s" % run_create_multiallelics_file_job_fn,
        ),
     file=fo)
fo.close()

get_ipython().system('bash {submit_create_multiallelics_file_jobs_fn}')

fo = open(short_contigs_fofn, 'w')
for chrom in sorted(list(genome.keys())):
    if chrom.startswith('Transfer'):
        vcf_fn = "%s/vcf/Pv_30_%s.final.vcf.gz" % (output_dir, chrom)
        print(vcf_fn, file=fo)
fo.close()

short_contigs_fofn

get_ipython().system('bcftools concat --file-list {short_contigs_fofn} -Oz -o {"%s/vcf/Pv_30_PvP01_00.final.vcf.gz" % output_dir}')

get_ipython().system('cp {GENOME_FN} {genome_fn}')
genome = pyfasta.Fasta(genome_fn)
genome

sorted(list(genome.keys()))

genome_length = 0
for chrom in genome.keys():
    genome_length += len(genome[chrom])
genome_length

transfer_length = 0
for chrom in genome.keys():
    if chrom.startswith('Transfer'):
        transfer_length += len(genome[chrom])
transfer_length

4802351/29052596

vrpipe_columns = [
    'path', 'sample', 'study', 'bases_of_1X_coverage', 'bases_of_2X_coverage', 'bases_of_5X_coverage',
    'mean_coverage', 'mean_insert_size', 'sd_insert_size', 'avg_read_length', 'bases_callable_percent',
    'bases_no_coverage_percent', 'bases_low_coverage_percent', 'bases_excessive_coverage_percent',
    'bases_poor_mapping_quality_percent', 'bases_ref_n_percent', 'reads', 'reads_mapped', 'reads_mapped_and_paired',
    'reads_properly_paired', 'reads_qc_failed', 'pairs_on_different_chromosomes', 'non_primary_alignments',
    'center_name'
]
print(",".join(vrpipe_columns[1:]))

metadata_columns = [
    'sample', 'study', 'center_name', 'bases_callable_proportion', 'bases_no_coverage_proportion', 'bases_low_coverage_proportion',
#     'bases_excessive_coverage_proportion', 'bases_poor_mapping_quality_proportion', 'bases_ref_n_proportion',
    'bases_poor_mapping_quality_proportion',
    'proportion_genome_covered_at_1x', 'proportion_genome_covered_at_5x', 'mean_coverage',
    'mean_insert_size', 'sd_insert_size', 'avg_read_length', 
    'reads_mapped_proportion', 'mapped_reads_properly_paired_proportion', 'pairs_on_different_chromosomes_proportion',
    'non_primary_alignments_proportion',
]

get_ipython().system('vrpipe-fileinfo --setup pv_30_mergelanes --metadata {",".join(vrpipe_columns[1:])} > {vrpipe_metadata_fn}')

# | grep '\.summary' \

get_ipython().system('vrpipe-fileinfo --setup pf_60_mergelanes --metadata sample,study,bases_of_1X_coverage,bases_of_2X_coverage,bases_of_5X_coverage,mean_coverage,mean_insert_size,sd_insert_size,avg_read_length,bases_callable_percent,bases_no_coverage_percent,bases_low_coverage_percent,bases_excessive_coverage_percent,bases_poor_mapping_quality_percent,bases_ref_n_percent,reads,reads_mapped,reads_mapped_and_paired,reads_properly_paired,reads_qc_failed,pairs_on_different_chromosomes,non_primary_alignments,center_name > {vrpipe_metadata_fn}')

# | grep '\.summary' \
# | sort -k 2,2 \

vcf_samples = vcf.Reader(filename='%s/vcf/Pv_30_PvP01_01_v1.final.vcf.gz' % output_dir).samples
print(len(vcf_samples))
vcf_samples[0:10]

tbl_vcf_samples = etl.fromcolumns([vcf_samples]).setheader(['sample'])
print(len(tbl_vcf_samples.data()))

tbl_vcf_samples.duplicates('sample')

metadata_columns

tbl_sample_metadata = (
    etl
    .fromtsv(vrpipe_metadata_fn)
    .setheader(vrpipe_columns)
#     .select(lambda rec: 'pe' in rec['path'] or rec['sample'] == 'PN0002-C')
    .convertnumbers()
#     .convert('avg_read_length', lambda val: val+1)
    .addfield('bases_callable_proportion', lambda rec: 0.0 if rec['bases_callable_percent'] == 'unknown' else round(rec['bases_callable_percent'] / 100, 4))
    .addfield('bases_no_coverage_proportion', lambda rec: 0.0 if rec['bases_no_coverage_percent'] == 'unknown' else round(rec['bases_no_coverage_percent'] / 100, 4))
    .addfield('bases_low_coverage_proportion', lambda rec: 0.0 if rec['bases_low_coverage_percent'] == 'unknown' else round(rec['bases_low_coverage_percent'] / 100, 4))
    .addfield('bases_excessive_coverage_proportion', lambda rec: 0.0 if rec['bases_excessive_coverage_percent'] == 'unknown' else round(rec['bases_excessive_coverage_percent'] / 100, 4))
    .addfield('bases_poor_mapping_quality_proportion', lambda rec: 0.0 if rec['bases_poor_mapping_quality_percent'] == 'unknown' else round(rec['bases_poor_mapping_quality_percent'] / 100, 4))
    .addfield('bases_ref_n_proportion', lambda rec: 0.0 if rec['bases_ref_n_percent'] == 'unknown' else round(rec['bases_ref_n_percent'] / 100, 4))
    .addfield('proportion_genome_covered_at_1x', lambda rec: 0.0 if rec['bases_of_1X_coverage'] == 'unknown' else round(rec['bases_of_1X_coverage'] / genome_length, 4))
    .addfield('proportion_genome_covered_at_5x', lambda rec: 0.0 if rec['bases_of_5X_coverage'] == 'unknown' else round(rec['bases_of_5X_coverage'] / genome_length, 4))
    .addfield('reads_mapped_proportion', lambda rec: 0.0 if rec['reads_mapped'] == 'unknown' else round(rec['reads_mapped'] / rec['reads'], 4))
    .addfield('mapped_reads_properly_paired_proportion', lambda rec: 0.0 if rec['reads_mapped'] == 'unknown' else round(rec['reads_properly_paired'] / rec['reads_mapped'], 4))
    # Note in the following we use reads_properly_paired/2 to get numbers of pairs of reads
    .addfield('pairs_on_different_chromosomes_proportion', lambda rec: 0.0 if rec['pairs_on_different_chromosomes'] == 'unknown' or rec['pairs_on_different_chromosomes'] == 0.0 else round(rec['pairs_on_different_chromosomes'] / (rec['pairs_on_different_chromosomes'] + ( rec['reads_properly_paired'] / 2)), 4))
    .addfield('non_primary_alignments_proportion', lambda rec: 0.0 if rec['reads_mapped'] == 'unknown' else round(rec['non_primary_alignments'] / rec['reads_mapped'], 4))
    .addfield('reads_qc_failed_proportion', lambda rec: 0.0 if rec['reads_qc_failed'] == 'unknown' else round(rec['reads_qc_failed'] / rec['reads'], 4))
    #  .leftjoin(tbl_solaris_metadata, lkey='sample', rkey='ox_code')
    #  .convert('run_accessions', 'NULL', where=lambda rec: rec['study'] == '1156-PV-ID-PRICE') # These were wrongly accessioned and are currently being removed from ENA
    #  .cut(['sample', 'study', 'src_code', 'run_accessions', 'genome_covered_at_1x', 'genome_covered_at_5x',
    #        'mean_coverage', 'avg_read_length'])
#     .cut(metadata_columns)
    .selectin('sample', vcf_samples)
    .sort('sample')
)
print(len(tbl_sample_metadata.data()))
tbl_sample_metadata.display(index_header=True)

tbl_vcf_samples.antijoin(tbl_sample_metadata, key='sample')

tbl_sample_metadata.duplicates('sample').displayall()

print(len(tbl_sample_metadata.selectin('sample', vcf_samples).data()))

tbl_sample_metadata.valuecounts('avg_read_length').sort('avg_read_length').displayall()

tbl_sample_metadata.valuecounts('study').sort('study').displayall()

studies = tbl_sample_metadata.distinct('study').values('study').array()
studies

study_vcf_jobs_manifest = '%s/study_vcf_jobs_manifest.txt' % output_dir
fo = open(study_vcf_jobs_manifest, 'w')
for study in studies:
    sample_ids = ",".join(tbl_sample_metadata.selecteq('study', study).values('sample'))
    for chrom in sorted(genome.keys()):
        if chrom.startswith('PvP01'):
            print('%s\t%s\t%s' % (study, chrom, sample_ids), file=fo)
    print('%s\t%s\t%s' % (study, 'PvP01_00', sample_ids), file=fo)
fo.close()

fo = open(create_study_vcf_job_fn, 'w')
print('''STUDY_VCF_JOBS_FILE=%s
 
JOB=$LSB_JOBINDEX
# JOB=19
 
IN=`sed "$JOB q;d" $STUDY_VCF_JOBS_FILE`
read -a LINE <<< "$IN"
STUDY=${LINE[0]}
CHROM=${LINE[1]}
SAMPLES=${LINE[2]}

OUTPUT_DIR=%s

mkdir -p $OUTPUT_DIR/study_vcfs/$STUDY

INPUT_VCF_FN=$OUTPUT_DIR/vcf/Pv_30_$CHROM.final.vcf.gz
OUTPUT_VCF_FN=$OUTPUT_DIR/study_vcfs/$STUDY/Pv_30__$STUDY\__$CHROM.vcf.gz

echo $OUTPUT_VCF_FN
echo $STUDY

bcftools view --samples $SAMPLES --output-file $OUTPUT_VCF_FN --output-type z $INPUT_VCF_FN
bcftools index --tbi $OUTPUT_VCF_FN
md5sum $OUTPUT_VCF_FN > $OUTPUT_VCF_FN.md5

''' % (
        study_vcf_jobs_manifest,
        output_dir,
        )
        , file=fo)
fo.close()

get_ipython().system('bash {create_study_vcf_job_fn}')

QUEUE = 'normal'
wc_output = get_ipython().getoutput('wc -l {study_vcf_jobs_manifest}')
NUM_JOBS = wc_output[0].split(' ')[0]
MEMORY = 8000
LOG_DIR = "%s/log" % output_dir

print(NUM_JOBS, LOG_DIR)

get_ipython().system('bsub -q {QUEUE} -G malaria-dk -J "s_vcf[1-{NUM_JOBS}]" -n2 -R"select[mem>{MEMORY}] rusage[mem={MEMORY}] span[hosts=1]" -M {MEMORY} -o {LOG_DIR}/output_%J-%I.log bash {create_study_vcf_job_fn}')

get_ipython().system('cp {output_dir}/vcf/* {nfs_final_vcf_dir}/')

get_ipython().system('cp -R {output_dir}/study_vcfs/* {nfs_release_dir}/')

for study in studies:
    get_ipython().system('cp /lustre/scratch116/malaria/pvivax/resources/gatk/pvp01_regions_20151213.bed.gz* {nfs_release_dir}/{study}/')

2+2



