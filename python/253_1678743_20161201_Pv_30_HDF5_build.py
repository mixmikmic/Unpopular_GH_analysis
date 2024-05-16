get_ipython().run_line_magic('run', '_standard_imports.ipynb')

output_dir = '/lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161201_Pv_30_HDF5_build'
vrpipe_fileinfo_fn = "%s/pv_30_genotype_gvcfs_200kb.txt" % output_dir
vcf_fofn = "%s/pv_30_genotype_gvcfs_20kb.fofn" % output_dir
vcf_stem = '/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pv3.0/20161130_Pv30_final_vcfs/vcf/Pv_30_{chrom}.final.vcf.gz'

nfs_release_dir = '/nfs/team112_internal/production/release_build/Pv/3_0_release_packages'
nfs_final_hdf5_dir = '%s/hdf5' % nfs_release_dir
get_ipython().system('mkdir -p {nfs_final_hdf5_dir}')

GENOME_FN = "/lustre/scratch109/malaria/pvivax/resources/gatk/PvivaxP01.genome.fasta"
genome_fn = "%s/PvivaxP01.genome.fasta" % output_dir

get_ipython().system('mkdir -p {output_dir}/hdf5')
get_ipython().system('mkdir -p {output_dir}/vcf')
get_ipython().system('mkdir -p {output_dir}/npy')
get_ipython().system('mkdir -p {output_dir}/scripts')
get_ipython().system('mkdir -p {output_dir}/log')

get_ipython().system('cp {GENOME_FN} {genome_fn}')

genome = pyfasta.Fasta(genome_fn)
genome

transfer_length = 0
for chrom in genome.keys():
    if chrom.startswith('Transfer'):
        transfer_length += len(genome[chrom])
transfer_length

fo = open("%s/scripts/vcfnp_variants.sh" % output_dir, 'w')
print('''#!/bin/bash

#set changes bash options
#x prints commands & args as they are executed
set -x
#-e  Exit immediately if a command exits with a non-zero status
set -e
#reports the last program to return a non-0 exit code rather than the exit code of the last problem
set -o pipefail

vcf=$1
chrom=$2

fasta=%s

vcf2npy \
    --vcf $vcf \
    --fasta $fasta \
    --output-dir %s/npy \
    --array-type variants \
    --task-size 20000 \
    --task-index $LSB_JOBINDEX \
    --progress 1000 \
    --chromosome $chrom \
    --arity ALT:6 \
    --arity AF:6 \
    --arity AC:6 \
    --arity svlen:6 \
    --dtype REF:a400 \
    --dtype ALT:a600 \
    --dtype MULTIALLELIC:a2 \
    --dtype RegionType:a25 \
    --dtype SNPEFF_AMINO_ACID_CHANGE:a105 \
    --dtype SNPEFF_CODON_CHANGE:a304 \
    --dtype SNPEFF_EFFECT:a33 \
    --dtype SNPEFF_EXON_ID:a2 \
    --dtype SNPEFF_FUNCTIONAL_CLASS:a8 \
    --dtype SNPEFF_GENE_NAME:a20 \
    --dtype SNPEFF_IMPACT:a8 \
    --dtype SNPEFF_TRANSCRIPT_ID:a20 \
    --dtype VARIANT_TYPE:a5 \
    --dtype VariantType:a40 \
    --exclude-field ID''' % (
        genome_fn,
        output_dir,
        )
        , file=fo)
fo.close()

fo = open("%s/scripts/vcfnp_calldata.sh" % output_dir, 'w')
print('''#!/bin/bash

set -x
set -e
set -o pipefail

vcf=$1
chrom=$2

fasta=%s

vcf2npy \
    --vcf $vcf \
    --fasta $fasta \
    --output-dir %s/npy \
    --array-type calldata_2d \
    --task-size 20000 \
    --task-index $LSB_JOBINDEX \
    --progress 1000 \
    --chromosome $chrom \
    --arity AD:7 \
    --arity PL:28 \
    --dtype PGT:a3 \
    --dtype PID:a12 \
    --exclude-field MIN_DP \
    --exclude-field RGQ \
    --exclude-field SB''' % (
        genome_fn,
        output_dir,
        )
        , file=fo)
fo.close()

fo = open("%s/scripts/vcfnp_concat.sh" % output_dir, 'w')
print('''#!/bin/bash

set -x
set -e
set -o pipefail

vcf=$1
outbase=$2
inputs=$3
output=${outbase}.h5

log=${output}.log

if [ -f ${output}.md5 ]
then
    echo $(date) skipping $chrom >> $log
else
    echo $(date) building $chrom > $log
    vcfnpy2hdf5 \
        --vcf $vcf \
        --input-dir $inputs \
        --output $output \
        --chunk-size 8388608 \
        --chunk-width 200 \
        --compression gzip \
        --compression-opts 1 \
        &>> $log
        
    md5sum $output > ${output}.md5 
fi''', file=fo)
fo.close()

task_size = 20000
for chrom in sorted(genome.keys()):
    vcf_fn = vcf_stem.format(chrom=chrom)
    n_tasks = '1-%s' % ((len(genome[chrom]) // task_size) + 1)
    print(chrom, n_tasks)

    task = "%s/scripts/vcfnp_variants.sh" % output_dir
    get_ipython().system('bsub -q normal -G malaria-dk -J "v_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')

    task = "%s/scripts/vcfnp_calldata.sh" % output_dir
    get_ipython().system('bsub -q normal -G malaria-dk -J "c_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')

fo = open("%s/scripts/vcfnp_variants_temp.sh" % output_dir, 'w')
print('''#!/bin/bash

#set changes bash options
#x prints commands & args as they are executed
set -x
#-e  Exit immediately if a command exits with a non-zero status
set -e
#reports the last program to return a non-0 exit code rather than the exit code of the last problem
set -o pipefail

vcf=$1
chrom=$2

fasta=%s

vcf2npy \
    --vcf $vcf \
    --fasta $fasta \
    --output-dir %s/npy_temp \
    --array-type variants \
    --task-size 20000 \
    --task-index $LSB_JOBINDEX \
    --progress 1000 \
    --chromosome $chrom \
    --arity ALT:6 \
    --arity AF:6 \
    --arity AC:6 \
    --arity svlen:6 \
    --dtype REF:a400 \
    --dtype ALT:a600 \
    --dtype MULTIALLELIC:a2 \
    --dtype RegionType:a25 \
    --dtype SNPEFF_AMINO_ACID_CHANGE:a105 \
    --dtype SNPEFF_CODON_CHANGE:a304 \
    --dtype SNPEFF_EFFECT:a33 \
    --dtype SNPEFF_EXON_ID:a2 \
    --dtype SNPEFF_FUNCTIONAL_CLASS:a8 \
    --dtype SNPEFF_GENE_NAME:a20 \
    --dtype SNPEFF_IMPACT:a8 \
    --dtype SNPEFF_TRANSCRIPT_ID:a20 \
    --dtype VARIANT_TYPE:a5 \
    --dtype VariantType:a40 \
    --exclude-field ID''' % (
        genome_fn,
        output_dir,
        )
        , file=fo)
fo.close()

# Three variants jobs from the above didn't complete. Killed them, then ran the following
get_ipython().system('mkdir -p {output_dir}/npy_temp')

task_size = 20000
for chrom in sorted(genome.keys()):
    vcf_fn = vcf_stem.format(chrom=chrom)
    n_tasks = '1-%s' % ((len(genome[chrom]) // task_size) + 1)
    print(chrom, n_tasks)

    task = "%s/scripts/vcfnp_variants_temp.sh" % output_dir
    get_ipython().system('bsub -q normal -G malaria-dk -J "v_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')

#     task = "%s/scripts/vcfnp_calldata.sh" % output_dir
#     !bsub -q normal -G malaria-dk -J "c_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} 

get_ipython().system('mv {output_dir}/npy_temp/v*.npy {output_dir}/npy/')

task = "%s/scripts/vcfnp_concat.sh" % output_dir
get_ipython().system('bsub -q long -G malaria-dk -J "hdf" -n8 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J.log bash {task} {vcf_stem.format(chrom=\'PvP01_01_v1\')} {output_dir}/hdf5/Pv_30 {output_dir}/npy')

output_dir

task = "%s/scripts/vcfnp_concat.sh" % output_dir
get_ipython().system('bsub -q long -G malaria-dk -J "full" -R"select[mem>16000] rusage[mem=16000] span[hosts=1]" -M 16000     -o {output_dir}/log/output_%J.log bash {task} {vcf_stem.format(chrom=\'Pf3D7_01_v3\')}     {output_dir}/hdf5/Pf_60     /lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161124_HDF5_build/npy')

y = h5py.File('%s/hdf5/Pf_60_npy_no_PID_PGT.h5' % output_dir, 'r')

(etl.wrap(
    np.unique(y['variants']['SNPEFF_EFFECT'], return_counts=True)
)
    .transpose()
    .pushheader('SNPEFF_EFFECT', 'number')
    .sort('number', reverse=True)
    .displayall()
)

task_size = 20000
for chrom in ['PvP01_00'] + sorted(genome.keys()):
    if chrom.startswith('Pv'):
        vcf_fn = vcf_stem.format(chrom=chrom)
        if chrom == 'PvP01_00':
            chrom_length = transfer_length
        else:
            chrom_length = len(genome[chrom])
        n_tasks = '1-%s' % ((chrom_length // task_size) + 1)
        print(chrom, n_tasks)

        task = "%s/scripts/vcfnp_variants.sh" % output_dir
        get_ipython().system('bsub -q normal -G malaria-dk -J "v_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')

        task = "%s/scripts/vcfnp_calldata.sh" % output_dir
        get_ipython().system('bsub -q normal -G malaria-dk -J "c_{chrom[6:8]}[{n_tasks}]" -n2 -R"select[mem>32000] rusage[mem=32000] span[hosts=1]" -M 32000 -o {output_dir}/log/output_%J-%I.log bash {task} {vcf_stem.format(chrom=chrom)} {chrom} ')





(etl.wrap(
    np.unique(y['variants']['CDS'], return_counts=True)
)
    .transpose()
    .pushheader('CDS', 'number')
    .sort('number', reverse=True)
    .displayall()
)

CDS = y['variants']['CDS'][:]
SNPEFF_EFFECT = y['variants']['SNPEFF_EFFECT'][:]
SNP = (y['variants']['VARIANT_TYPE'][:] == b'SNP')
INDEL = (y['variants']['VARIANT_TYPE'][:] == b'INDEL')

np.unique(CDS[SNP], return_counts=True)

2+2

y['variants']['VARIANT_TYPE']

pd.value_counts(INDEL)

pd.crosstab(SNPEFF_EFFECT[SNP], CDS[SNP])

2+2

df = pd.DataFrame({'CDS': CDS, 'SNPEFF_EFFECT':SNPEFF_EFFECT})

writer = pd.ExcelWriter("/nfs/users/nfs_r/rp7/SNPEFF_for_Rob.xlsx")
pd.crosstab(SNPEFF_EFFECT, CDS).to_excel(writer)
writer.save()



pd.crosstab(SNPEFF_EFFECT, y['variants']['CHROM'])

np.unique(y['variants']['svlen'], return_counts=True)

y = h5py.File('%s/hdf5/Pf_60_npy_no_PID_PGT_10pc.h5.h5' % output_dir, 'r')
y

# for field in y['variants'].keys():
for field in ['svlen']:
    print(field, np.unique(y['variants'][field], return_counts=True))









get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_no_PID_PGT_10pc     --output {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5 > {output_dir}/hdf5/Pf_60_no_PID_PGT_10pc.h5.md5 ')





get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_subset_1pc     --output {output_dir}/hdf5/Pf_60_subset_1pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_subset_1pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_subset_1pc.h5 > {output_dir}/hdf5/Pf_60_subset_1pc.h5.md5 ')

get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_subset     --output {output_dir}/hdf5/Pf_60_subset_10pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_subset_10pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_subset_10pc.h5 > {output_dir}/hdf5/Pf_60_subset_10pc.h5.md5 ')

get_ipython().system('vcfnpy2hdf5     --vcf {vcf_fn}     --input-dir {output_dir}/npy_subset     --output {output_dir}/hdf5/Pf_60_subset_10pc.h5     --chunk-size 8388608     --chunk-width 200     --compression gzip     --compression-opts 1     &>> {output_dir}/hdf5/Pf_60_subset_10pc.h5.log')

get_ipython().system('md5sum {output_dir}/hdf5/Pf_60_subset_10pc.h5 > {output_dir}/hdf5/Pf_60_subset_10pc.h5.md5 ')

get_ipython().system('{output_dir}/scripts/vcfnp_concat.sh {vcf_fn} {output_dir}/hdf5/Pf_60')

fo = open("%s/scripts/vcfnp_concat.sh" % output_dir, 'w')
print('''#!/bin/bash

set -x
set -e
set -o pipefail

vcf=$1
outbase=$2
# inputs=${vcf}.vcfnp_cache
inputs=%s/npy
output=${outbase}.h5

log=${output}.log

if [ -f ${output}.md5 ]
then
    echo $(date) skipping $chrom >> $log
else
    echo $(date) building $chrom > $log
    vcfnpy2hdf5 \
        --vcf $vcf \
        --input-dir $inputs \
        --output $output \
        --chunk-size 8388608 \
        --chunk-width 200 \
        --compression gzip \
        --compression-opts 1 \
        &>> $log
        
    md5sum $output > ${output}.md5 
fi''' % (
        output_dir,
        )
      , file=fo)
fo.close()

#     nv=$(ls -1 ${inputs}/v* | wc -l)
#     nc=$(ls -1 ${inputs}/c* | wc -l)
#     echo variants files $nv >> $log
#     echo calldata files $nc >> $log
#     if [ "$nv" -ne "$nc" ]
#     then
#         echo missing npy files
#         exit 1
#     fi

get_ipython().system('cp {output_dir}/hdf5/Pv_30.h5 {nfs_final_hdf5_dir}/')
get_ipython().system('cp {output_dir}/hdf5/Pv_30.h5.md5 {nfs_final_hdf5_dir}/')



