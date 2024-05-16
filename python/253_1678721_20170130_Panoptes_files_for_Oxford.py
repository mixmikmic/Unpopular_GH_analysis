get_ipython().run_line_magic('run', '_standard_imports.ipynb')

panoptes_final_metadata_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/pf_6_0_panoptes_samples_final_20170124.txt.gz"
sample_6_0_fn = "/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_sample_metadata.txt"

hdf_fn = "/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/hdf5/Pf_60.h5"
gff_fn = "/lustre/scratch116/malaria/pfalciparum/resources/snpEff/data/Pfalciparum_GeneDB_Oct2016/Pfalciparum.noseq.gff3"
genome_fn = "/lustre/scratch116/malaria/pfalciparum/resources/Pfalciparum.genome.fasta"
panoptes_samples_fn = "/nfs/team112_internal/production/release_build/Pf/6_0_release_packages/Pf_60_samples_panoptes_20170130.txt"

tbl_panoptes_samples_final = (
    etl
    .fromtsv(panoptes_final_metadata_fn)
    .cutout('pc_pass_missing')
    .cutout('pc_genome_covered_at_1x')
    .cutout('sort_AlfrescoStudyCode')
)
print(len(tbl_panoptes_samples_final.data()))
print(len(tbl_panoptes_samples_final.distinct('Sample').data()))
tbl_panoptes_samples_final

tbl_sample_6_0 = (
    etl
    .fromtsv(sample_6_0_fn)
    .cutout('study')
    .cutout('source_code')
    .cutout('run_accessions')
)
print(len(tbl_sample_6_0.data()))
print(len(tbl_sample_6_0.distinct('sample').data()))
tbl_sample_6_0

tbl_panoptes_samples = (
    tbl_panoptes_samples_final
    .join(tbl_sample_6_0, lkey='Sample', rkey='sample')
)
print(len(tbl_panoptes_samples.data()))
print(len(tbl_panoptes_samples.distinct('Sample').data()))
tbl_panoptes_samples

len(tbl_panoptes_samples.header())

callset = h5py.File(hdf_fn, mode='r')
callset['samples'][:]

v_decode_ascii = np.vectorize(lambda x: x.decode('ascii'))

sample_concordance = (v_decode_ascii(callset['samples'][:]) == tbl_panoptes_samples.values('Sample').array())

np.unique(sample_concordance, return_counts=True)

np.all(v_decode_ascii(callset['samples'][:]) == tbl_panoptes_samples.values('Sample').array())

tbl_panoptes_samples.totsv(panoptes_samples_fn, lineterminator='\n')

tbl_panoptes_samples

