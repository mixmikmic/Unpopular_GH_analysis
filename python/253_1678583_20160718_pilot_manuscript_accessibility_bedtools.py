get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')

# see 20160525_CallableLoci_bed_release_5.ipynb
lustre_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5"
callable_loci_bed_fn_format = "%s/results/callable_loci_%%s.bed" % lustre_dir

plot_dir = "/nfs/team112_internal/rp7/data/pf3k/analysis/20160718_pilot_manuscript_accessibility"
get_ipython().system('mkdir -p {plot_dir}')

core_regions_fn = "%s/core_regions_20130225.bed" % lustre_dir

callable_loci_fn = "%s/callable_loci_high_coverage_samples.bed" % plot_dir
callable_loci_merged_fn = "%s/callable_loci_merged_samples.bed" % plot_dir

multiIntersectBed = '/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/multiIntersectBed'
bedtools = '/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools'

# core_regions_fn = '/nfs/team112_internal/rp7/src/github/malariagen/pf-crosses/meta/regions-20130225.bed.gz'

core_genome_dict = collections.OrderedDict()
for chrom in ['Pf3D7_%02d_v3' % i for i in range(1, 15)]:
    this_chrom_regions = (etl
                          .fromtabix(REGIONS_FN, chrom)
                          .pushheader('chrom', 'start', 'end', 'region')
                          .convertnumbers()
                          )
    chrom_length = np.max(this_chrom_regions.convert('end', int).values('end').array())
    core_genome_dict[chrom] = np.zeros(chrom_length, dtype=bool)
    for rec in this_chrom_regions:
        if rec[3] == 'Core':
            core_genome_dict[chrom][rec[1]:rec[2]] = True

tbl_sample_metadata = etl.fromtsv(SAMPLE_METADATA_FN)

tbl_field_samples = tbl_sample_metadata.select(lambda rec: not rec['study'] in ['1041', '1042', '1043', '1104', ''])

len(tbl_field_samples.data())

def count_symbol(i=1):
    if i%10 == 0:
        return(str((i//10)*10))
    else:
        return('.')

# ox_codes = tbl_field_samples_extended.selectge('core_bases_callable', core_genome_length*0.95).values('sample').array(dtype='U12')
# len(ox_codes)

ox_codes = tbl_field_samples.values('sample').array(dtype='U12')
len(ox_codes)

ox_codes.dtype

callable_loci_merged_fn = '/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_2512_field_merged.bed'
for i, ox_code in enumerate(ox_codes):
    print('%s' % count_symbol(i), end='', flush=True)
    callable_loci_bed_fn = callable_loci_bed_fn_format % ox_code
    get_ipython().system('grep CALLABLE {callable_loci_bed_fn} >> {callable_loci_merged_fn}')

get_ipython().system("sort -T /lustre/scratch111/malaria/rp7/temp -k1,1 -k2,2n {callable_loci_merged_fn} > {callable_loci_merged_fn.replace('.bed', '.sort.bed')}")

# !/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools genomecov \
# -i {callable_loci_merged_fn.replace('.bed', '.sort.bed')} \
# -g {GENOME_FN+'.fai'} -bga \
# > /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_2512_field_merged_coverage.bed

get_ipython().system("/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools genomecov -i {callable_loci_merged_fn.replace('.bed', '.sort.bed')} -g {GENOME_FN+'.fai'} -d > /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_2512_field_merged_coverage.txt")

# !bgzip -f /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.bed
get_ipython().system('bgzip -f /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_2512_field_merged_coverage.txt')

# !tabix -f -p bed /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.bed.gz
get_ipython().system('tabix -f -s 1 -b 2 -e 2 /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_2512_field_merged_coverage.txt.gz')

merged_coverage_fn = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_2512_field_merged_coverage.txt.gz"

accessibility_array = (etl
 .fromtsv(merged_coverage_fn)
 .pushheader(['chrom', 'pos', 'coverage'])
 .convertnumbers()
 .toarray(dtype='a11, i4, i4')
)

print(len(accessibility_array))
print(accessibility_array[0])
accessibility_array

accessibility_array_fn = "%s/accessibility_array_2512_field.npy" % plot_dir
np.save(accessibility_array_fn, accessibility_array)

del(accessibility_array)
gc.collect()

accessibility_array_fn = "%s/accessibility_array_2512_field.npy" % plot_dir
accessibility_array = np.load(accessibility_array_fn)

accessibility_colors = {
    'Core': 'white',
    'SubtelomericHypervariable': 'red',
    'InternalHypervariable': 'orange',
    'SubtelomericRepeat': 'brown',
    'Centromere': 'black'
#     'InternalHypervariable': '#b20000',
}

def plot_accessibility(accessibility=accessibility_array, callset='2512_field', bin_size=1000, number_of_samples = 2512):

    fig = plt.figure(figsize=(11.69*1, 8.27*1))
    gs = GridSpec(2*14, 1, height_ratios=([1.0, 1.0])*14)
    gs.update(hspace=0, left=.12, right=.98, top=.98, bottom=.02)

    print('\n', bin_size)
    for i in range(14):
        print(i+1, end=" ")
        chrom = 'Pf3D7_%02d_v3' % (i + 1)
        pos = accessibility[accessibility['chrom']==chrom.encode('ascii')]['pos']
        coverage = accessibility[accessibility['chrom']==chrom.encode('ascii')]['coverage']
        max_pos = np.max(pos)
        if bin_size == 1:
            binned_coverage, bin_centres = coverage, pos
        else:
            binned_coverage, bins, _ = scipy.stats.binned_statistic(pos, coverage, bins=np.arange(1, max_pos, bin_size))
            bin_centres = (bins[:-1]+bins[1:]) / 2
        ax = fig.add_subplot(gs[i*2])
        ax.plot(bin_centres, binned_coverage/number_of_samples)
    #     ax.plot(pos, coverage/number_of_samples)
        ax.set_xlim(0, 3300000)
        ax.set_xticks(range(0, len(core_genome_dict[chrom]), 100000))
        ax.set_xticklabels(np.arange(0, len(core_genome_dict[chrom])/1e+6, 0.1))
        tbl_regions = (etl
            .fromtabix(REGIONS_FN, chrom)
            .pushheader('chrom', 'start', 'end', 'region')
            .convertnumbers()
        )
        for region_chrom, start_pos, end_pos, region_type in tbl_regions.data():
            if region_type != 'Core':
                ax.axvspan(start_pos, end_pos, facecolor=accessibility_colors[region_type], alpha=0.1)
        for s in 'left', 'right', 'top':
            ax.spines[s].set_visible(False)
    #         ax.set_yticklabels([])
        ax.get_xaxis().tick_bottom()
        ax.set_yticks([])

        ax.set_ylabel(i+1, rotation='horizontal', horizontalalignment='right', verticalalignment='center')

        ax.set_xlabel('')
        if i < 13:
            ax.set_xticklabels([])
    #     ax.spines['top'].set_bounds(0, len(core_genome_dict[chrom]))    
        ax.spines['bottom'].set_bounds(0, len(core_genome_dict[chrom]))
    
    fig.savefig(os.path.join(plot_dir, 'short_read_accesibility_%s_%dbp_windows.png' % (callset, bin_size)), dpi=150)
    fig.savefig(os.path.join(plot_dir, 'short_read_accesibility_%s_%dbp_windows.pdf' % (callset, bin_size)))

plot_accessibility()



ox_codes_5 = ['7G8', 'GB4', 'ERS740940', 'ERS740937', 'ERS740936']
callable_loci_merged_fn = '/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_5_validation_merged.bed'
for i, ox_code in enumerate(ox_codes_5):
    print('%s' % count_symbol(i), end='', flush=True)
    callable_loci_bed_fn = callable_loci_bed_fn_format % ox_code
    get_ipython().system('grep CALLABLE {callable_loci_bed_fn} >> {callable_loci_merged_fn}')

get_ipython().system("sort -T /lustre/scratch111/malaria/rp7/temp -k1,1 -k2,2n {callable_loci_merged_fn} > {callable_loci_merged_fn.replace('.bed', '.sort.bed')}")

get_ipython().system("/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools genomecov -i {callable_loci_merged_fn.replace('.bed', '.sort.bed')} -g {GENOME_FN+'.fai'} -d > /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_5_validation_merged_coverage.txt")

get_ipython().system('bgzip -f /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_5_validation_merged_coverage.txt')
get_ipython().system('tabix -f -s 1 -b 2 -e 2 /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_5_validation_merged_coverage.txt.gz')

merged_coverage_fn = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_5_validation_merged_coverage.txt.gz"

accessibility_array_5_validation = (etl
 .fromtsv(merged_coverage_fn)
 .pushheader(['chrom', 'pos', 'coverage'])
 .convertnumbers()
 .toarray(dtype='a11, i4, i4')
)

print(len(accessibility_array_5_validation))
print(accessibility_array_5_validation[0])
accessibility_array_5_validation

accessibility_array_fn = "%s/accessibility_array_5_validation.npy" % plot_dir
np.save(accessibility_array_fn, accessibility_array_5_validation)

plot_accessibility(accessibility=accessibility_array_5_validation, callset='5_validation')

plot_dir



