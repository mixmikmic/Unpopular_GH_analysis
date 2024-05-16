archive_dir = '/nfs/team112_internal/rp7/data/Pf/WillH_1'
for chrom in ['Pf3D7_%02d_v3' % n for n in range(1, 15)] + ['Pf3D7_API_v3', 'Pf_M76611']:
    get_ipython().system('mkdir -p {"%s/vcf/vcf_symlinks/%s" % (archive_dir, chrom)}')

get_ipython().system('cp -R /lustre/scratch109/malaria/WillH_1/meta {archive_dir}/')

