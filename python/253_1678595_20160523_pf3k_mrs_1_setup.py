get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')

RELEASE_DIR = "%s/mrs_1" % DATA_DIR
RESOURCES_DIR = '%s/resources' % RELEASE_DIR

# GENOME_FN = "/nfs/pathogen003/tdo/Pfalciparum/3D7/Reference/Oct2011/Pf3D7_v3.fasta" # Note this ref used by Thomas is different to other refs we have used, e.g. chromsomes aren't in numerical order
GENOME_FN = "/lustre/scratch109/malaria/pf3k_methods/resources/Pfalciparum.genome.fasta"
SNPEFF_DIR = "/lustre/scratch109/malaria/pf3k_methods/resources/snpEff"
REGIONS_FN = "/nfs/team112_internal/rp7/src/github/malariagen/pf-crosses/meta/regions-20130225.bed.gz"

RELEASE_METADATA_FN = "%s/pf3k_mrs_1_sample_metadata.txt" % RELEASE_DIR
WG_VCF_FN = "%s/vcf/pf3k_mrs_1.vcf.gz" % RELEASE_DIR

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'

print(WG_VCF_FN)

chromosomes = ["Pf3D7_%02d_v3" % x for x in range(1, 15, 1)] + [
    'Pf3D7_API_v3', 'Pf_M76611'
]
chromosome_vcfs = ["%s/vcf/SNP_INDEL_%s.combined.filtered.vcf.gz" % (RELEASE_DIR, x) for x in chromosomes]

if not os.path.exists(RESOURCES_DIR):
    os.makedirs(RESOURCES_DIR)

get_ipython().system('cp {GENOME_FN}* {RESOURCES_DIR}')
get_ipython().system('cp -R {SNPEFF_DIR} {RESOURCES_DIR}')
get_ipython().system('cp -R {REGIONS_FN} {RESOURCES_DIR}')

for lustre_dir in ['input', 'output', 'meta']:
    os.makedirs("/lustre/scratch109/malaria/pf3k_mrs_1/%s" % lustre_dir)



