get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')

release5_final_files_dir = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0'
wg_vcf_fn = "%s/SNP_INDEL_WG.combined.filtered.vcf.gz" % (release5_final_files_dir)

output_dir = '/nfs/team112_internal/rp7/data/pf3k/analysis/20160802_5PC_MAF_for_Zam'
get_ipython().system('mkdir -p {output_dir}')
output_fn = "%s/SNP_INDEL_WG.combined.filtered.maf_ge_5pc.vcf.gz" % output_dir

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'

get_ipython().system('{BCFTOOLS} view --include \'MAF[0]>=0.05 & FILTER="PASS" & ALT[1]!="*"\' --output-file {output_fn} --output-type z {wg_vcf_fn}')
get_ipython().system('{BCFTOOLS} index -f --tbi {output_fn}')

# --regions Pf3D7_01_v3:100000-110000 \



2+2



