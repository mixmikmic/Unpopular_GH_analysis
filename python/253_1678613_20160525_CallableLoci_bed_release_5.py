get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')

import stat
from sh import ssh
bsub = sh.Command('bsub')

output_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5"
bam_fofn = "%s/pf3k_sample_bams.txt" % output_dir
get_ipython().system('mkdir -p {output_dir}/scripts')
get_ipython().system('mkdir -p {output_dir}/results')
get_ipython().system('mkdir -p {output_dir}/logs')

get_ipython().system('vrpipe-fileinfo --setup pf3kgatk_mergelanes --step 4 --display tab --metadata sample > {bam_fofn}')

GenomeAnalysisTK="/software/jre1.7.0_25/bin/java -Xmx4G -jar /nfs/team112_internal/production/tools/bin/gatk/GenomeAnalysisTK-3.5/GenomeAnalysisTK.jar"

GENOME_FN

tbl_bams = etl.fromtsv(bam_fofn)
print(len(tbl_bams.data()))
tbl_bams

for bam_fn, sample in tbl_bams.data():
    print('.', sep='')
    bed_fn = "%s/results/callable_loci_%s.bed" % (output_dir, sample)
    summary_fn = "%s/results/summary_table_%s.txt" % (output_dir, sample)

    if not os.path.exists(bed_fn):
#     if True:
        script_fn = "%s/scripts/CallableLoci_%s.sh" % (output_dir, sample)
        fo = open(script_fn, 'w')
        print('''%s -T CallableLoci -R %s -I %s -summary %s -o %s
''' % (
                GenomeAnalysisTK,
                GENOME_FN,
                bam_fn,
                summary_fn,
                bed_fn,
            ),
            file = fo
        )
        fo.close()
        st = os.stat(script_fn)
        os.chmod(script_fn, st.st_mode | stat.S_IEXEC)
        bsub(
            '-G', 'malaria-dk',
            '-P', 'malaria-dk',
            '-q', 'normal',
            '-o', '%s/logs/CL_%s.out' % (output_dir, sample),
            '-e', '%s/logs/CL_%s.err' % (output_dir, sample),
            '-J', 'CL_%s' % (sample),
            '-R', "'select[mem>8000] rusage[mem=8000]'",
            '-M', '8000',
            script_fn)

2+2

