get_ipython().run_line_magic('run', 'imports_mac.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')

get_ipython().system('mkdir -p {os.path.dirname(INTERIM5_VCF_FOFN)}')
get_ipython().system('rsync -av malsrv2:{INTERIM5_VCF_FOFN} {os.path.dirname(INTERIM5_VCF_FOFN)}')

for release in CHROM_VCF_FNS.keys():
    for chrom in CHROM_VCF_FNS[release].keys():
        vcf_fn = CHROM_VCF_FNS[release][chrom]
        get_ipython().system('mkdir -p {os.path.dirname(vcf_fn)}')
        get_ipython().system('rsync -avL malsrv2:{vcf_fn} {os.path.dirname(vcf_fn)}')

vcf_fn = WG_VCF_FNS['release3']
get_ipython().system('mkdir -p {os.path.dirname(vcf_fn)}')
get_ipython().system('rsync -avL malsrv2:{vcf_fn} {os.path.dirname(vcf_fn)}')

RELEASE_3_VCF_FN = '/nfs/team112_internal/production/release_build/Pf3K/pilot_3_0/all_merged_with_calls_vfp_v4.vcf.gz'
RELEASE_4_DIR = '/nfs/team112_internal/production/release_build/Pf3K/pilot_4_0'
INTERIM_5_VCF_FOFN = '/lustre/scratch109/malaria/pf3k_methods/input/output_fofn/pf3kgatk_variant_filtration_ps583for_2640samples.output'



