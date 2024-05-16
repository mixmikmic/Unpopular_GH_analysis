get_ipython().run_line_magic('run', '_standard_imports.ipynb')

from Bio.Seq import Seq

diff_breakpoint_samples = ['PH0243-C', 'PH0247-C', 'PH0484-C', 'PH0906-C', 'PH0912-C']
mdr1_samples = ['PH0254-C']

# Note we used same samples as were previously used for HRP deletions. See 20160621_HRP_sample_metadata.ipynb
manifest_5_0_fn = '/nfs/team112_internal/rp7/data/Pf/hrp/metadata/hrp_manifest_20160621.txt'
# Note the following file created whilst 6.0 build still in progress, so don't have final sample bams
manifest_6_0_fn = '/nfs/team112_internal/rp7/data/Pf/6_0/metadata/plasmepsin_manifest_6_0_20160922.txt'
scratch_dir = '/lustre/scratch109/malaria/rp7/data/ppq'

genome_fn = '/lustre/scratch116/malaria/pfalciparum/resources/Pfalciparum.genome.fasta'

genome = SeqIO.to_dict(SeqIO.parse(genome_fn, "fasta"))
# , key_function=get_accession)

genome.keys()

genome['Pf3D7_14_v3'].count()

tbl_manifest = etl.fromtsv(manifest_fn)
print(len(tbl_manifest.data()))
tbl_manifest

tbl_bams_5_0 = tbl_manifest.selectin('sample', diff_breakpoint_samples)
tbl_bams_5_0.display()

tbl_manifest.selectin('sample', mdr1_samples)

genome['Pf3D7_14_v3'][283032:283070].seq

genome['Pf3D7_14_v3'].seq.count(genome['Pf3D7_14_v3'][283020:283069].seq)

temp = list(genome.keys())
temp.sort()
temp

sorted(genome.keys())

import re
for chrom in sorted(genome.keys()):
#     print(chrom, genome[chrom].seq.count(genome['Pf3D7_14_v3'][283027:283069].seq))
#     print(chrom, genome[chrom].seq.find('agaagtaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'))
    print(chrom, [m.start() for m in re.finditer('agaagtaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', str(genome[chrom].seq))])

print()

for chrom in sorted(genome.keys()):
    print(chrom, [m.start() for m in re.finditer('aaaaaaagaagtaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', str(genome[chrom].seq))])

print()

for chrom in sorted(genome.keys()):
    print(chrom, [m.start() for m in re.finditer('aaaaagaagtaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', str(genome[chrom].seq))])

    



for chrom in sorted(genome.keys()):
    print(chrom, [m.start() for m in re.finditer('aaatgaagggaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', str(genome[chrom].seq))])

breakpoint_sequences = collections.OrderedDict()
breakpoint_sequences['0bp'] = 'GATAATCACAC'
breakpoint_sequences['1bp'] = 'CGATAATCACACT'
breakpoint_sequences['5bp'] = 'ATTACGATAATCACACTGTTG'
breakpoint_sequences['10bp'] = 'TTATGATTACGATAATCACACTGTTGGTTTC'
breakpoint_sequences['15bp'] = 'ACCGTTTATGATTACGATAATCACACTGTTGGTTTCGCCCT'
breakpoint_sequences['20bp'] = 'ATTTTACCGTTTATGATTACGATAATCACACTGTTGGTTTCGCCCTTGCCA'

breakpoint_sequences['0bp'].lower()

for breakpoint_sequence in [x.lower() for x in breakpoint_sequences.values()]:
    print(breakpoint_sequence)
    for chrom in sorted(genome.keys()):
        print(chrom, [m.start() for m in re.finditer(breakpoint_sequence, str(genome[chrom].seq))])

for breakpoint_sequence in ['cgataatcacact', 'cgataatcacac', 'gataatcacact', 'gataatcacac']:
    print(breakpoint_sequence)
    for chrom in sorted(genome.keys()):
        print(chrom, [m.start() for m in re.finditer(breakpoint_sequence, str(genome[chrom].seq))])

genome['Pf3D7_14_v3'][283027:283069].seq



for breakpoint_sequence in ['aaaaaaaaaaaaaaaaaaaa']:
    print(breakpoint_sequence)
    for chrom in ['Pf3D7_05_v3']:
#     for chrom in sorted(genome.keys()):
        print(chrom, [m.start() for m in re.finditer(breakpoint_sequence, str(genome[chrom].seq))])

print("mkdir -p /lustre/scratch109/malaria/rp7/data/ppq/bams/5_0/")
for rec in tbl_bams_5_0.data():
    original_bam = rec[0]
    macbook_bam = "%s/bams/5_0/%s.bam" % (scratch_dir, rec[1])
    print("scp malsrv2:%s %s" % (original_bam.replace('.bam', '.bam.bai'), macbook_bam.replace('.bam', '.bam.bai')))
    print("scp malsrv2:%s %s" % (original_bam, macbook_bam))

get_ipython().system('mkdir -p {os.path.dirname(manifest_6_0_fn)}')
get_ipython().system('vrpipe-fileinfo --setup pf_60_bqsr --metadata sample > {manifest_6_0_fn}')

tbl_manifest_6_0 = etl.fromtsv(manifest_6_0_fn).select(lambda rec: rec[0][-3:] == 'bam')
print(len(tbl_manifest_6_0.data()))
tbl_manifest_6_0

tbl_bams_6_0 = tbl_manifest_6_0.selectin('sample', diff_breakpoint_samples).sort('sample')
tbl_bams_6_0.display()

print("mkdir -p /lustre/scratch109/malaria/rp7/data/ppq/bams/6_0/")
for rec in tbl_bams_6_0.data():
    original_bam = rec[0]
    macbook_bam = "%s/bams/6_0/%s.bam" % (scratch_dir, rec[1])
    print("scp malsrv2:%s %s" % (original_bam.replace('.bam', '.bai'), macbook_bam.replace('.bam', '.bai')))
    print("scp malsrv2:%s %s" % (original_bam, macbook_bam))

tbl_manifest_6_0.selectin('sample', mdr1_samples)



import stat
bsub = sh.Command('bsub')

for breakpoint_sequence_name in breakpoint_sequences:
# for breakpoint_sequence_name in ['10bp']:
    breakpoint_reads_dir = '%s/plasmepsin_1_3_7979_samples/%s' % (output_dir, breakpoint_sequence_name)
    get_ipython().system('mkdir -p {breakpoint_reads_dir}/results')
    get_ipython().system('mkdir -p {breakpoint_reads_dir}/scripts')
    get_ipython().system('mkdir -p {breakpoint_reads_dir}/logs')

    breakpoint_sequence = "'%s|%s'" % (
        breakpoint_sequences[breakpoint_sequence_name],
        Seq(breakpoint_sequences[breakpoint_sequence_name]).reverse_complement()
    )
    for rec in tbl_manifest:
        bam_fn = rec[0]
        ox_code = rec[1]
        print('.', end='')
        num_breakpoint_reads_fn = "%s/results/num_breakpoint_reads_%s.txt" % (breakpoint_reads_dir, ox_code)
        if not os.path.exists(num_breakpoint_reads_fn):
            script_fn = "%s/scripts/nbpr_%s.sh" % (breakpoint_reads_dir, ox_code)
            fo = open(script_fn, 'w')
            print('samtools view %s | grep -E %s | wc -l > %s' % (bam_fn, breakpoint_sequence, num_breakpoint_reads_fn), file = fo)
            fo.close()
            st = os.stat(script_fn)
            os.chmod(script_fn, st.st_mode | stat.S_IEXEC)
            bsub(
                '-G', 'malaria-dk',
                '-P', 'malaria-dk',
                '-q', 'normal',
                '-o', '%s/logs/nbpr_%s.out' % (breakpoint_reads_dir, ox_code),
                '-e', '%s/logs/nbpr_%s.err' % (breakpoint_reads_dir, ox_code),
                '-J', 'nbpr_%s' % (ox_code),
                '-R', "'select[mem>4000] rusage[mem=4000]'",
                '-M', '4000',
                script_fn
            )

tbl_manifest



pf_5_0_breakpoint_reads = collections.OrderedDict()

# for breakpoint_sequence_name in breakpoint_sequences:
for breakpoint_sequence_name in ['10bp']:
    pf_5_0_breakpoint_reads[breakpoint_sequence_name] = collections.OrderedDict()
    breakpoint_reads_dir = '%s/plasmepsin_1_3_pf_5_0/%s' % (output_dir, breakpoint_sequence_name)
    for rec in tbl_5_0_manifest:
        bam_fn = rec[0]
        ox_code = rec[2]
        print('.', end='')
        num_breakpoint_reads_fn = "%s/results/num_breakpoint_reads_%s.txt" % (breakpoint_reads_dir, ox_code)
        fi = open(num_breakpoint_reads_fn, 'r')
        pf_5_0_breakpoint_reads[breakpoint_sequence_name][ox_code] = int(fi.read())

breakpoint_reads_dict = collections.OrderedDict()

for breakpoint_sequence_name in breakpoint_sequences:
    print(breakpoint_sequence_name)
# for breakpoint_sequence_name in ['10bp']:
    breakpoint_reads_dict[breakpoint_sequence_name] = collections.OrderedDict()
    breakpoint_reads_dir = '%s/plasmepsin_1_3_7979_samples/%s' % (output_dir, breakpoint_sequence_name)
    for rec in tbl_manifest.data():
        bam_fn = rec[0]
        ox_code = rec[1]
        print('.', end='')
        num_breakpoint_reads_fn = "%s/results/num_breakpoint_reads_%s.txt" % (breakpoint_reads_dir, ox_code)
        fi = open(num_breakpoint_reads_fn, 'r')
        breakpoint_reads_dict[breakpoint_sequence_name][ox_code] = int(fi.read())

tbl_breakpoint_reads = (etl.wrap(zip(breakpoint_reads_dict['0bp'].keys(), breakpoint_reads_dict['0bp'].values())).pushheader(['ox_code', 'bp_reads_0bp'])
 .join(
        etl.wrap(zip(breakpoint_reads_dict['1bp'].keys(), breakpoint_reads_dict['1bp'].values())).pushheader(['ox_code', 'bp_reads_1bp']),
        key='ox_code')
 .join(
        etl.wrap(zip(breakpoint_reads_dict['5bp'].keys(), breakpoint_reads_dict['5bp'].values())).pushheader(['ox_code', 'bp_reads_5bp']),
        key='ox_code')
 .join(
        etl.wrap(zip(breakpoint_reads_dict['10bp'].keys(), breakpoint_reads_dict['10bp'].values())).pushheader(['ox_code', 'bp_reads_10bp']),
        key='ox_code')
 .join(
        etl.wrap(zip(breakpoint_reads_dict['15bp'].keys(), breakpoint_reads_dict['15bp'].values())).pushheader(['ox_code', 'bp_reads_15bp']),
        key='ox_code')
 .join(
        etl.wrap(zip(breakpoint_reads_dict['20bp'].keys(), breakpoint_reads_dict['20bp'].values())).pushheader(['ox_code', 'bp_reads_20bp']),
        key='ox_code')
)
# tbl_breakpoint_reads.displayall()

tbl_breakpoint_reads.selectgt('bp_reads_10bp', 0).displayall()

tbl_breakpoint_reads.totsv("%s/plasmepsin_1_3_7979_samples.tsv" % output_dir)

tbl_breakpoint_reads.toxlsx("%s/plasmepsin_1_3_7979_samples.xlsx" % output_dir)

2+2



