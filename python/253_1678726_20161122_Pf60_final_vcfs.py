get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')

output_dir = '/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161122_Pf60_final_vcfs'
vrpipe_vcfs_dir = '/nfs/team112_internal/production_files/Pf/6_0'

gff_fn = "/lustre/scratch116/malaria/pfalciparum/resources/snpEff/data/Pfalciparum_GeneDB_Oct2016/Pfalciparum.noseq.gff3"
cds_gff_fn = "%s/gff/Pfalciparum_GeneDB_Oct2016.Pfalciparum.noseq.gff3.cds.gz" % output_dir
annotations_header_fn = "%s/intermediate_files/annotations.hdr" % (output_dir)

run_create_multiallelics_file_job_fn = "%s/scripts/run_create_multiallelics_file_job.sh" % output_dir
submit_create_multiallelics_file_jobs_fn = "%s/scripts/submit_create_multiallelics_file_jobs.sh" % output_dir


GENOME_FN = "/lustre/scratch116/malaria/pfalciparum/resources/Pfalciparum.genome.fasta"
BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'

get_ipython().system('mkdir -p {output_dir}/gff')
get_ipython().system('mkdir -p {output_dir}/vcf')
get_ipython().system('mkdir -p {output_dir}/intermediate_files')
get_ipython().system('mkdir -p {output_dir}/tables')
get_ipython().system('mkdir -p {output_dir}/scripts')
get_ipython().system('mkdir -p {output_dir}/log')

cds_gff_fn

run_create_multiallelics_file_job_fn

get_ipython().system("grep CDS {gff_fn} | sort -k1,1 -k4n,5n | cut -f 1,4,5 | sed 's/$/\\t1/' | bgzip -c > {cds_gff_fn} && tabix -s1 -b2 -e3 {cds_gff_fn}")

fo=open(annotations_header_fn, 'w')
print('##INFO=<ID=CDS,Number=0,Type=Flag,Description="Is position coding">', file=fo)
print('##INFO=<ID=VARIANT_TYPE,Number=1,Type=String,Description="SNP or indel (INDEL)">', file=fo)
print('##INFO=<ID=MULTIALLELIC,Number=1,Type=String,Description="Is position biallelic (BI), biallelic plus spanning deletion (SD) or truly multiallelic (MU)">', file=fo)
fo.close()

fo = open(run_create_multiallelics_file_job_fn, 'w')
print('''FASTA_FAI_FILE=%s.fai
 
# JOB=$LSB_JOBINDEX
JOB=16
 
IN=`sed "$JOB q;d" $FASTA_FAI_FILE`
read -a LINE <<< "$IN"
CHROM=${LINE[0]}

INPUT_SITES_VCF_FN=%s/SNP_INDEL_$CHROM.combined.filtered.vcf.gz
INPUT_FULL_VCF_FN=%s/vcf/SNP_INDEL_$CHROM.combined.filtered.vcf.gz
MULTIALLELIC_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.multiallelic.txt
SNPS_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.snps.txt.gz
INDELS_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.indels.txt.gz
ANNOTATION_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.annotated.vcf.gz
NORMALISED_ANNOTATION_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.annotated.normalised.vcf.gz
OUTPUT_VCF_FN=%s/vcf/Pf_60_$CHROM.final.vcf.gz

# echo $INPUT_VCF_FN
# echo $OUTPUT_TXT_FN
 
/nfs/users/nfs_r/rp7/anaconda3/bin/python /nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\ 6.0/scripts/create_multiallelics_file.py \
-i $INPUT_SITES_VCF_FN -o $MULTIALLELIC_FN

bgzip -f $MULTIALLELIC_FN && tabix -s1 -b2 -e2 $MULTIALLELIC_FN.gz

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\tSNP\n' --include 'TYPE="snp"' $INPUT_SITES_VCF_FN | \
bgzip -c > $SNPS_FN && tabix -s1 -b2 -e2 -f $SNPS_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\tINDEL\n' --include 'TYPE!="snp"' $INPUT_SITES_VCF_FN | \
bgzip -c > $INDELS_FN && tabix -s1 -b2 -e2 -f $INDELS_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a %s -c CHROM,FROM,TO,CDS -h %s $INPUT_SITES_VCF_FN | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $SNPS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $INDELS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $MULTIALLELIC_FN.gz -c CHROM,POS,REF,ALT,INFO/MULTIALLELIC \
--remove ^INFO/AC,INFO/AN,INFO/QD,INFO/MQ,INFO/FS,INFO/SOR,INFO/DP,INFO/VariantType,INFO/VQSLOD,INFO/RegionType,\
INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EFFECT,INFO/SNPEFF_EXON_ID,\
INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,\
INFO/CDS,INFO/VARIANT_TYPE,INFO/MULTIALLELIC \
-Oz -o $ANNOTATION_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools index --tbi \
$ANNOTATION_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools norm \
-m -any --fasta-ref %s $INPUT_SITES_VCF_FN | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools view \
--include 'ALT!="*"' | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-h %s \
-a $ANNOTATION_FN -c CDS,VARIANT_TYPE,MULTIALLELIC \
--include 'INFO/AC>0' \
--remove ^INFO/AC,INFO/AN,INFO/AF,INFO/VQSLOD -Oz -o $NORMALISED_ANNOTATION_FN

# /nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools index --tbi \
# $NORMALISED_ANNOTATION_FN

''' % (
        GENOME_FN,
        vrpipe_vcfs_dir,
        vrpipe_vcfs_dir,
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
        )
        , file=fo)
fo.close()

fo = open(run_create_multiallelics_file_job_fn, 'w')
print('''FASTA_FAI_FILE=%s.fai
 
JOB=$LSB_JOBINDEX
# JOB=16
 
IN=`sed "$JOB q;d" $FASTA_FAI_FILE`
read -a LINE <<< "$IN"
CHROM=${LINE[0]}

INPUT_SITES_VCF_FN=%s/SNP_INDEL_$CHROM.combined.filtered.vcf.gz
INPUT_FULL_VCF_FN=%s/vcf/SNP_INDEL_$CHROM.combined.filtered.vcf.gz
MULTIALLELIC_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.multiallelic.txt
SNPS_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.snps.txt.gz
INDELS_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.indels.txt.gz
ANNOTATION_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.annotated.txt.gz
NORMALISED_ANNOTATION_FN=%s/intermediate_files/SNP_INDEL_$CHROM.combined.filtered.annotated.normalised.vcf.gz
OUTPUT_VCF_FN=%s/vcf/Pf_60_$CHROM.final.vcf.gz

# echo $INPUT_VCF_FN
# echo $OUTPUT_TXT_FN
 
/nfs/users/nfs_r/rp7/anaconda3/bin/python /nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\ 6.0/scripts/create_multiallelics_file.py \
-i $INPUT_SITES_VCF_FN -o $MULTIALLELIC_FN

bgzip -f $MULTIALLELIC_FN && tabix -s1 -b2 -e2 $MULTIALLELIC_FN.gz

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\tSNP\n' --include 'TYPE="snp"' $INPUT_SITES_VCF_FN | \
bgzip -c > $SNPS_FN && tabix -s1 -b2 -e2 -f $SNPS_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\tINDEL\n' --include 'TYPE!="snp"' $INPUT_SITES_VCF_FN | \
bgzip -c > $INDELS_FN && tabix -s1 -b2 -e2 -f $INDELS_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a %s -c CHROM,FROM,TO,CDS -h %s $INPUT_SITES_VCF_FN | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $SNPS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $INDELS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $MULTIALLELIC_FN.gz -c CHROM,POS,REF,ALT,INFO/MULTIALLELIC | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools query \
-f'%%CHROM\t%%POS\t%%REF\t%%ALT\t%%CDS\t%%VARIANT_TYPE\t%%MULTIALLELIC\n' | \
bgzip -c > $ANNOTATION_FN

tabix -s1 -b2 -e2 $ANNOTATION_FN

#/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools index --tbi \
#$ANNOTATION_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools norm \
-m -any --fasta-ref %s $INPUT_SITES_VCF_FN | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools view \
--include 'ALT!="*"' | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-h %s \
-a $ANNOTATION_FN -c CHROM,POS,REF,ALT,CDS,VARIANT_TYPE,MULTIALLELIC \
--include 'INFO/AC>0' \
--remove ^INFO/AC,INFO/AN,INFO/AF,INFO/VQSLOD -Oz -o $NORMALISED_ANNOTATION_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools index --tbi \
$NORMALISED_ANNOTATION_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a %s -c CHROM,FROM,TO,CDS -h %s $INPUT_FULL_VCF_FN | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $SNPS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $INDELS_FN -c CHROM,POS,REF,ALT,INFO/VARIANT_TYPE | \
/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools annotate \
-a $MULTIALLELIC_FN.gz -c CHROM,POS,REF,ALT,INFO/MULTIALLELIC \
--remove ^INFO/AC,INFO/AN,INFO/QD,INFO/MQ,INFO/FS,INFO/SOR,INFO/DP,INFO/VariantType,INFO/VQSLOD,INFO/RegionType,\
INFO/SNPEFF_AMINO_ACID_CHANGE,INFO/SNPEFF_CODON_CHANGE,INFO/SNPEFF_EFFECT,INFO/SNPEFF_EXON_ID,\
INFO/SNPEFF_FUNCTIONAL_CLASS,INFO/SNPEFF_GENE_NAME,INFO/SNPEFF_IMPACT,INFO/SNPEFF_TRANSCRIPT_ID,\
INFO/CDS,INFO/VARIANT_TYPE,INFO/MULTIALLELIC,FORMAT/BCS \
-Oz -o $OUTPUT_VCF_FN

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools index --tbi \
$OUTPUT_VCF_FN

''' % (
        GENOME_FN,
        vrpipe_vcfs_dir,
        vrpipe_vcfs_dir,
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

fo = open(submit_create_multiallelics_file_jobs_fn, 'w')
print('''FASTA_FAI_FILE=%s.fai
LOG_DIR=%s/log
 
NUM_CHROMS=`wc -l < $FASTA_FAI_FILE`
QUEUE=long

bsub -q $QUEUE -G malaria-dk -J "ma[1-$NUM_CHROMS]" -R"select[mem>2000] rusage[mem=2000] span[hosts=1]" -M 2000 -o $LOG_DIR/output_%%J-%%I.log %s
''' % (
        GENOME_FN,
        output_dir,
        "bash %s" % run_create_multiallelics_file_job_fn,
        ),
     file=fo)
fo.close()

get_ipython().system('bash {submit_create_multiallelics_file_jobs_fn}')



