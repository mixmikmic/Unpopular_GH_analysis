get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')

BCFTOOLS = '/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/pf3k_techbm/opt_4/bcftools/bcftools'
five_strain_vcf_fn = '/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160718_Thomas_5_validation_vcf/SNP_INDEL_WG.for_thomas.vcf.gz'

output_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160719_sanity_check_Thomas_numbers"
get_ipython().system('mkdir -p {output_dir}')

vcf_7G8 = "%s/7G8.vcf.gz" % output_dir
get_ipython().system('{BCFTOOLS} view -s 7G8 {five_strain_vcf_fn} | {BCFTOOLS} view --include \'AC>0 && ALT!="*"\' -Oz -o {vcf_7G8}')
get_ipython().system('{BCFTOOLS} index --tbi {vcf_7G8}')

6319+33433

6319+33433+4381

6319+33433+4381-909



