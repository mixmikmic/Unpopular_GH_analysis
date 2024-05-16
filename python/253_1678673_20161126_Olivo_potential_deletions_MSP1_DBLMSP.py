output_dir = '/lustre/scratch109/malaria/rp7/data/methods-dev/20161126_Olivo_potential_deletions_MSP1_DBLMSP'
get_ipython().system('mkdir -p {output_dir}')

pf_6_bams_fn = '/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161117_run_Olivo_GRC/pf_60_mergelanes.txt'
pf_5_bams_fn = '/nfs/team112_internal/rp7/data/Pf/hrp/metadata/hrp_manifest_20160621.txt'

get_ipython().system('grep PA0169 {pf_6_bams_fn}')

get_ipython().system('grep PA0169 {pf_5_bams_fn}')

get_ipython().system('grep PM0293 {pf_6_bams_fn}')

get_ipython().system('grep PM0293 {pf_5_bams_fn}')

get_ipython().system('samtools view -b /lustre/scratch116/malaria/pfalciparum/output/6/0/2/a/37065/4_bam_mark_duplicates_v2/pe.1.markdup.bam Pf3D7_09_v3:1200000-1210000 > ~/PA0169_C_Pf3D7_09_v3_1200000_1210000_bwa_mem.bam')
get_ipython().system('samtools index ~/PA0169_C_Pf3D7_09_v3_1200000_1210000_bwa_mem.bam')

get_ipython().system('samtools view -b /lustre/scratch116/malaria/pfalciparum/output/6/0/2/a/37065/4_bam_mark_duplicates_v2/pe.1.markdup.bam Pf3D7_09_v3:1200000-1210000 > ~/PA0169_C_Pf3D7_09_v3_1200000_1210000_bwa_mem.bam')
get_ipython().system('samtools index ~/PA0169_C_Pf3D7_09_v3_1200000_1210000_bwa_mem.bam')

get_ipython().system('samtools view -b /lustre/scratch116/malaria/pfalciparum/output/6/0/2/a/37065/4_bam_mark_duplicates_v2/pe.1.markdup.bam Pf3D7_09_v3:1200000-1210000 > ~/PA0169_C_Pf3D7_09_v3_1200000_1210000.bam ')

get_ipython().system('samtools view -b /lustre/scratch116/malaria/pfalciparum/output/6/0/2/a/37065/4_bam_mark_duplicates_v2/pe.1.markdup.bam Pf3D7_10_v3:1403500-1420000 > ~/PA0169_C_Pf3D7_10_v3_1403500_1420000_bwa_mem.bam')
get_ipython().system('samtools index ~/PA0169_C_Pf3D7_10_v3_1403500_1420000_bwa_mem.bam')

get_ipython().system('samtools view -b /lustre/scratch116/malaria/pfalciparum/output/6/0/2/a/37065/4_bam_mark_duplicates_v2/pe.1.markdup.bam Pf3D7_09_v3:1200000-1210000 > ~/PA0169_C_Pf3D7_10_v3_1403500_1420000_bwa_aln.bam')
get_ipython().system('samtools index ~/PA0169_C_Pf3D7_10_v3_1403500_1420000_bwa_aln.bam')

ftp_dir = 'ftp://ftp.sanger.ac.uk/pub/project/pathogens/Plasmodium/falciparum/PF3K/PilotReferenceGenomes/GenomeSequence/Version1/'
get_ipython().system('wget -r {ftp_dir} -P {output_dir}/')

output_dir

get_ipython().system('samtools bamshuf -uon 128 /lustre/scratch116/malaria/pfalciparum/output/6/0/2/a/37065/4_bam_mark_duplicates_v2/pe.1.markdup.bam tmp-prefix | samtools bam2fq -s se.fq.gz - | bwa mem -p ref.fa -')

