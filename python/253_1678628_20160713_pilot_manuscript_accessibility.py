get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')

# see 20160525_CallableLoci_bed_release_5.ipynb
callable_loci_bed_fn_format = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_%s.bed"

plot_dir = "/nfs/team112_internal/rp7/data/pf3k/analysis/20160713_pilot_manuscript_accessibility"
get_ipython().system('mkdir -p {plot_dir}')

# core_regions_fn = '/nfs/team112_internal/rp7/src/github/malariagen/pf-crosses/meta/regions-20130225.bed.gz'

etl.fromtsv(REGIONS_FN).pushheader('chrom', 'start', 'end', 'region')

core_genome_dict = collections.OrderedDict()
for chrom in ['Pf3D7_%02d_v3' % i for i in range(1, 15)]:
    this_chrom_regions = (etl
                          .fromtabix(core_regions_fn, chrom)
                          .pushheader('chrom', 'start', 'end', 'region')
                          .convertnumbers()
                          )
    chrom_length = np.max(this_chrom_regions.convert('end', int).values('end').array())
    core_genome_dict[chrom] = np.zeros(chrom_length, dtype=bool)
    for rec in this_chrom_regions:
        if rec[3] == 'Core':
            core_genome_dict[chrom][rec[1]:rec[2]] = True

core_genome_length = 0
for chrom in core_genome_dict:
    print(chrom, len(core_genome_dict[chrom]), np.sum(core_genome_dict[chrom]))
    core_genome_length = core_genome_length + np.sum(core_genome_dict[chrom])
print(core_genome_length)

tbl_sample_metadata = etl.fromtsv(SAMPLE_METADATA_FN)

tbl_field_samples = tbl_sample_metadata.select(lambda rec: not rec['study'] in ['1041', '1042', '1043', '1104', ''])

len(tbl_field_samples.data())

bases_callable = collections.OrderedDict()
core_bases_callable = collections.OrderedDict()
autosomes = ['Pf3D7_%02d_v3' % i for i in range(1, 15)]
for i, ox_code in enumerate(tbl_field_samples.values('sample')):
#     print(i, ox_code)
    this_sample_callable_loci = collections.OrderedDict()
    callable_loci_bed_fn = callable_loci_bed_fn_format % ox_code
    for chrom in core_genome_dict.keys():
        chrom_length = len(core_genome_dict[chrom])
        this_sample_callable_loci[chrom] = np.zeros(chrom_length, dtype=bool)
    tbl_this_sample_callable_loci = (etl
                                     .fromtsv(callable_loci_bed_fn)
                                     .pushheader('chrom', 'start', 'end', 'region')
                                     .selecteq('region', 'CALLABLE')
                                     .selectin('chrom', autosomes)
                                     .convertnumbers()
                                    )
    for rec in tbl_this_sample_callable_loci.data():
        this_sample_callable_loci[rec[0]][rec[1]:rec[2]] = True
    bases_callable[ox_code] = 0
    core_bases_callable[ox_code] = 0
    for chrom in core_genome_dict.keys():
        bases_callable[ox_code] = bases_callable[ox_code] + np.sum(this_sample_callable_loci[chrom])
        core_bases_callable[ox_code] = core_bases_callable[ox_code] + np.sum((this_sample_callable_loci[chrom] & core_genome_dict[chrom]))
#     print(ox_code, bases_callable, core_bases_callable)
#     print(i, type(i))
    print('%d' % (i%10), end='', flush=True)
    
        

20296931 / 20782107 

20782107 * 0.95

proportion_core_callable = np.array(core_bases_callable.values())

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1, 1, 1)
ax.hist(proportion_core_callable, bins=np.linspace(0.0, 1.0, num=101))
fig.tight_layout()
fig.savefig("%s/proportion_core_callable_histogram.pdf" % plot_dir)

