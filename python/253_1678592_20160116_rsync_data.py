get_ipython().run_line_magic('run', 'imports_mac.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')

get_ipython().system('mkdir -p {os.path.dirname(INTERIM5_VCF_FOFN)}')
get_ipython().system('rsync -avL malsrv2:{INTERIM5_VCF_FOFN} {os.path.dirname(INTERIM5_VCF_FOFN)}')

for release in CHROM_VCF_FNS.keys():
    for chrom in CHROM_VCF_FNS[release].keys():
        vcf_fn = CHROM_VCF_FNS[release][chrom]
        get_ipython().system('mkdir -p {os.path.dirname(vcf_fn)}')
        get_ipython().system('rsync -avL malsrv2:{vcf_fn} {os.path.dirname(vcf_fn)}')

vcf_fn = WG_VCF_FNS['release3']
get_ipython().system('mkdir -p {os.path.dirname(vcf_fn)}')
get_ipython().system('rsync -avL malsrv2:{vcf_fn} {os.path.dirname(vcf_fn)}')

if not os.path.exists(RELEASE4_RESOURCES_DIR):
    get_ipython().system('mkdir -p {RELEASE4_RESOURCES_DIR}')
get_ipython().system('rsync -avL malsrv2:{RELEASE4_RESOURCES_DIR} {os.path.dirname(RELEASE4_RESOURCES_DIR)}')

# GATK executables
get_ipython().system('mkdir -p /nfs/team112_internal/production/tools/bin')
get_ipython().system('rsync -avL malsrv2:/nfs/team112_internal/production/tools/bin/gatk /nfs/team112_internal/production/tools/bin/')

# Other executables - decided to leave these for now
# !rsync -avL malsrv2:/nfs/team112_internal/production/tools/bin /nfs/team112_internal/production/tools/

