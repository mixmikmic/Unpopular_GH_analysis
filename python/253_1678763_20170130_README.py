WG_VCF_FN = "/nfs/team112_internal/production_files/Pf/1147_Conway/conway_5_1_annot_gt.vcf.gz"
FINAL_VCF_FN = "/nfs/team112_internal/production_files/Pf/1147_Conway/conway_5_1_annot_gt_nonref.vcf.gz"
# BCFTOOLS = 'bcftools'

get_ipython().system('tabix -p vcf {WG_VCF_FN}')
get_ipython().system('md5sum {WG_VCF_FN} > {WG_VCF_FN}.md5')

get_ipython().system("bcftools view --include 'AC>0' --output-type z --output-file {FINAL_VCF_FN} {WG_VCF_FN}")
get_ipython().system('bcftools index --tbi {FINAL_VCF_FN}')
get_ipython().system('md5sum {FINAL_VCF_FN} > {FINAL_VCF_FN}.md5')

number_of_variants = get_ipython().getoutput("bcftools query -f '%CHROM\\t%POS\\n' {FINAL_VCF_FN} | wc -l")
number_of_snps = get_ipython().getoutput('bcftools query -f \'%CHROM\\t%POS\\n\' --include \'TYPE="snp"\' {FINAL_VCF_FN} | wc -l')
number_of_ref_only = get_ipython().getoutput("bcftools query -f '%CHROM\\t%POS\\n' --include 'N_ALT=0' {FINAL_VCF_FN} | wc -l")
number_of_pass_variants = get_ipython().getoutput('bcftools query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS"\' {FINAL_VCF_FN} | wc -l')
number_of_pass_snps = get_ipython().getoutput('bcftools query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE="snp"\' {FINAL_VCF_FN} | wc -l')
number_of_pass_ref_only = get_ipython().getoutput('bcftools query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && N_ALT=0\' {FINAL_VCF_FN} | wc -l')
number_of_pass_biallelic_variants = get_ipython().getoutput('bcftools query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && N_ALT=1\' {FINAL_VCF_FN} | wc -l')
number_of_pass_biallelic_snps = get_ipython().getoutput('bcftools query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE="snp" && N_ALT=1\' {FINAL_VCF_FN} | wc -l')
number_of_pass_biallelic_ref_only = get_ipython().getoutput('bcftools query -f \'%CHROM\\t%POS\\n\' --include \'FILTER="PASS" && TYPE="snp" && N_ALT=1 && N_ALT=0\' {FINAL_VCF_FN} | wc -l')

print("%s variants" % ("{:,}".format(int(number_of_variants[0]))))
print("%s SNPs" % ("{:,}".format(int(number_of_snps[0]))))
print("%s ref only" % ("{:,}".format(int(number_of_ref_only[0]))))
print()
print("%s PASS variants" % ("{:,}".format(int(number_of_pass_variants[0]))))
print("%s PASS SNPs" % ("{:,}".format(int(number_of_pass_snps[0]))))
print("%s PASS ref only" % ("{:,}".format(int(number_of_pass_ref_only[0]))))
print()
print("%s PASS biallelic variants" % ("{:,}".format(int(number_of_pass_biallelic_variants[0]))))
print("%s PASS biallelic SNPs" % ("{:,}".format(int(number_of_pass_biallelic_snps[0]))))
print("%s PASS biallelic ref only" % ("{:,}".format(int(number_of_pass_biallelic_ref_only[0]))))
print()
     

number_of_pass_inc_noncoding_variants = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER!="Biallelic" && FILTER!="HetUniq" && FILTER!="HyperHet" && FILTER!="MaxCoverage" && FILTER!="MinAlt" && FILTER!="MinCoverage" && FILTER!="MonoAllelic" && FILTER!="NoAltAllele" && FILTER!="Region" && FILTER!="triallelic"\' {FINAL_VCF_FN} | wc -l')
number_of_pass_inc_noncoding_variants

number_of_hq_noncoding_variants = get_ipython().getoutput('{BCFTOOLS} query -f \'%CHROM\\t%POS\\n\' --include \'FILTER!="PASS" && FILTER!="Biallelic" && FILTER!="HetUniq" && FILTER!="HyperHet" && FILTER!="MaxCoverage" && FILTER!="MinAlt" && FILTER!="MinCoverage" && FILTER!="MonoAllelic" && FILTER!="NoAltAllele" && FILTER!="Region" && FILTER!="triallelic"\' {FINAL_VCF_FN} | wc -l')
number_of_hq_noncoding_variants

number_of_samples = get_ipython().getoutput('bcftools query --list-samples {FINAL_VCF_FN} | wc -l')
number_of_samples

print('''
===================================================================================
MalariaGEN P. falciparum Community Project - Biallelic SNP genotypes for study 1147
===================================================================================

Date: 2017-01-30


Description 
-----------

Through an analysis of 3,394 parasite samples collected at 42 different locations in Africa, Asia, America and Oceania, we identified single nucleotide polymorphisms (SNPs). This download includes genotyping data for samples contributed to the MalariaGEN Plasmodium falciparum Community Project under study 1147 DC-MRC-Mauritania that were genotyped at these SNPs.

Potential data users are asked to respect the legitimate interest of the Community Project and its partners by abiding any restrictions on the use of a data as described in the Terms of Use: http://www.malariagen.net/projects/parasite/pf/use-p-falciparum-community-project-data

For more information on the P. falciparum Community Project that generated these data, please visit: https://www.malariagen.net/projects/p-falciparum-community-project

Genotyping data is currently released for all identified biallelic single nucleotide polymorphisms (SNPs) that are segregating amongst the {number_of_samples} samples of study 1147. 

The methods used to generate the data are described in detail in MalariaGEN Plasmodium falciparum Community Project, eLife (2016), DOI: 10.7554/eLife.08714.

This data was created as an ad-hoc build and hasn't been quality assessed by the MalariaGEN team.


Citation information
--------------------

Publications using these data should acknowledge and cite the source of the data using the following format: "This publication uses data from the MalariaGEN Plasmodium falciparum Community Project as described in Genomic epidemiology of artemisinin resistant malaria, eLife, 2016 (DOI: 10.7554/eLife.08714)."


File descriptions
-----------------

- conway_5_1_annot_gt_nonref.vcf.gz

The data file ("*.vcf.gz") is a zipped VCF format file containing all samples in the study.  The file, once unzipped, is a tab-separated text file, but may be too big to open in Excel.  

The format is described in https://github.com/samtools/hts-specs

Tools to assist in handling VCF files are freely available from
https://vcftools.github.io/index.html
http://samtools.github.io/bcftools/

- conway_5_1_annot_gt_nonref.vcf.gz.tbi

This is a tabix index file for conway_5_1_annot_gt_nonref.vcf.gz

Further details on tabix indexes are available at
http://www.htslib.org/doc/tabix.html

- conway_5_1_annot_gt_nonref.vcf.gz.md5

This is an MD5 checksum for conway_5_1_annot_gt_nonref.vcf.gz


Contents of the VCF file
------------------------

The VCF file contains details of {number_of_variants} SNPs in {number_of_samples} samples. These are all the SNPs discovered in the MalariaGEN 5.0 release to partners that are segregating and biallelic in the {number_of_samples} samples.

It is important to note that many of these SNPs are considered low quality. Only the variants for which the FILTER column is set to PASS should be considered of high quality. There are {number_of_pass_variants} such high-quality PASS SNPs. Note that this set only includes coding SNPs (those in exons). There are an additional {number_of_hq_noncoding_variants} SNPs that are in non-coding regions but which pass all other variant filters.

Columns 10 and onwards of the VCF contain the information for each sample. The first component of this (GT) is the genotype call. A value of 0 indicates a homozygous reference call (at least 5 reads in total and <= 1 read with alternative allele). A value of 1 indicates a homozygous alternative call (at least 5 reads in total and <= 1 read with reference allele). A value of 0/1 indicates the sample has a heterozygous call (at least 5 reads in total, >=2 reads with reference allele and >=2 reads with alternative allele). A value of . indicates a missing genotype call (<5 reads in total).

'''.format(
        number_of_variants="{:,}".format(int(number_of_snps[0])),
        number_of_samples="{:,}".format(int(number_of_samples[0])),
        number_of_pass_variants="{:,}".format(int(number_of_pass_snps[0])),
        number_of_hq_noncoding_variants="{:,}".format(int(number_of_hq_noncoding_variants[0])),
    )
)



