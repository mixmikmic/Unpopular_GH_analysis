get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')

bam_fns = collections.OrderedDict()
bam_fns['PD0479-C Pf3k'] = '/lustre/scratch109/malaria/pf3k_methods/output/2/8/4/f/290788/4_bam_mark_duplicates_v2/pe.1.markdup.bam'
bam_fns['PD0471-C Pf3k'] = '/lustre/scratch109/malaria/pf3k_methods/output/8/3/4/7/290780/4_bam_mark_duplicates_v2/pe.1.markdup.bam'
bam_fns['PD0479-C Pf 5.0'] = '/lustre/scratch109/malaria/pfalciparum/output/4/4/3/3/43216/1_bam_merge/pe.1.bam'
bam_fns['PD0471-C Pf 5.0'] = '/lustre/scratch109/malaria/pfalciparum/output/f/5/2/7/43208/1_bam_merge/pe.1.bam'

vcf_fns = collections.OrderedDict()
vcf_fns['Pf3k'] = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/SNP_INDEL_WG.combined.filtered.vcf.gz'
vcf_fns['Pf 5.0'] = '/nfs/team112_internal/production_files/Pf/5_0/pf_50_vfp1.newCoverageFilters_pass_5_99.5.vcf.gz'

exon4_coordinates = 'Pf3D7_07_v3:404283-404415'

seeds = collections.OrderedDict()
seeds['Pf3D7_07_v3:404290-404310'] = 'TTATACAATTATCTCGGAGCA'
seeds['Pf3D7_07_v3:404356-404376'] = 'TTTGAAACACAAGAAGAAAAT'

from Bio.Seq import Seq
for seed in seeds:
    print(seed)
    print(Seq(seeds[seed]).reverse_complement())

for vcf_fn in vcf_fns:
    print(vcf_fn)
    get_ipython().system('tabix {vcf_fns[vcf_fn]} {exon4_coordinates} | cut -f 1-7')

for vcf_fn in vcf_fns:
    print(vcf_fn)
    get_ipython().system('tabix {vcf_fns[vcf_fn]} {exon4_coordinates} | grep -v MinAlt | cut -f 1-7')

for seed in seeds:
    for bam_fn in bam_fns:
        print('\n\n', seed, bam_fn)
        seed_sequence = "'%s|%s'" % (
            seeds[seed],
            Seq(seeds[seed]).reverse_complement()
        )
        get_ipython().system('samtools view {bam_fns[bam_fn]} | grep -E {seed_sequence} | cut -f 1-9')
    

for seed in seeds:
    for bam_fn in bam_fns:
        print('\n\n', seed, bam_fn)
        seed_sequence = "'%s|%s'" % (
            seeds[seed],
            Seq(seeds[seed]).reverse_complement()
        )
        get_ipython().system('samtools view -f 4 {bam_fns[bam_fn]} | grep -E {seed_sequence} | cut -f 1-9')
    



