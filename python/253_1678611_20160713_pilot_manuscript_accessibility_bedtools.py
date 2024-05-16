get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')

# see 20160525_CallableLoci_bed_release_5.ipynb
lustre_dir = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5"
callable_loci_bed_fn_format = "%s/results/callable_loci_%%s.bed" % lustre_dir

plot_dir = "/nfs/team112_internal/rp7/data/pf3k/analysis/20160713_pilot_manuscript_accessibility"
get_ipython().system('mkdir -p {plot_dir}')

core_regions_fn = "%s/core_regions_20130225.bed" % lustre_dir

callable_loci_fn = "%s/callable_loci_high_coverage_samples.bed" % plot_dir
callable_loci_merged_fn = "%s/callable_loci_merged_samples.bed" % plot_dir

multiIntersectBed = '/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/multiIntersectBed'
bedtools = '/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools'

# core_regions_fn = '/nfs/team112_internal/rp7/src/github/malariagen/pf-crosses/meta/regions-20130225.bed.gz'

core_regions_fn

REGIONS_FN

get_ipython().system('zgrep Core {REGIONS_FN} > {core_regions_fn}')

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

core_genome_length = 0
for chrom in core_genome_dict:
    print(chrom, len(core_genome_dict[chrom]), np.sum(core_genome_dict[chrom]))
    core_genome_length = core_genome_length + np.sum(core_genome_dict[chrom])
print(core_genome_length)

tbl_sample_metadata = etl.fromtsv(SAMPLE_METADATA_FN)

tbl_field_samples = tbl_sample_metadata.select(lambda rec: not rec['study'] in ['1041', '1042', '1043', '1104', ''])

len(tbl_field_samples.data())

for sample in tbl_field_samples.values('sample'):
    print('.', end='')
    callable_loci_bed_fn = "%s/results/callable_loci_%s.bed" % (lustre_dir, sample)
    core_bases_callable_fn = "%s/results/core_bases_callable_%s.txt" % (lustre_dir, sample)

    if not os.path.exists(core_bases_callable_fn):
        script_fn = "%s/scripts/core_bases_callable_%s.sh" % (lustre_dir, sample)
        fo = open(script_fn, 'w')
        print('''grep CALLABLE %s | %s intersect -a - -b %s | %s genomecov -i - -g %s | grep -P 'genome\t1' | cut -f 3 > %s
''' % (
                callable_loci_bed_fn,
                bedtools,
                core_regions_fn,
                bedtools,
                GENOME_FN+'.fai',
                core_bases_callable_fn,
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
            '-o', '%s/logs/CL_%s.out' % (lustre_dir, sample),
            '-e', '%s/logs/CL_%s.err' % (lustre_dir, sample),
            '-J', 'CBC_%s' % (sample),
            '-R', "'select[mem>1000] rusage[mem=1000]'",
            '-M', '1000',
            script_fn)

def read_core_base_callable_file(sample='PF0249-C'):
    core_bases_callable_fn = "%s/results/core_bases_callable_%s.txt" % (lustre_dir, sample)
    with open(core_bases_callable_fn, 'r') as f:
        bases_callable = f.readline()
        if bases_callable == '':
            return(0)
        else:
            return(int(bases_callable))
   

read_core_base_callable_file()

tbl_field_samples_extended = tbl_field_samples.addfield('core_bases_callable', lambda rec: read_core_base_callable_file(rec[0]))
tbl_field_samples_extended.cut(['sample', 'core_bases_callable'])

len(tbl_field_samples.data())

ox_codes = tbl_field_samples.values('sample').array()

def calc_callable_core(callable_loci_bed_fn='/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_PA0007-C.bed'):
    core_bases_callable_fn = callable_loci_bed_fn.replace('.bed', '.core_callable.txt')
    get_ipython().system("grep CALLABLE {callable_loci_bed_fn} |     {bedtools} intersect -a - -b {core_regions_fn} |     {bedtools} genomecov -i - -g {GENOME_FN+'.fai'} |     grep -P 'genome\\t1' |     cut -f 3 > {core_bases_callable_fn}")

calc_callable_core()

callable_loci_bed_fns = [callable_loci_bed_fn_format % ox_code for ox_code in ox_codes]
print(len(callable_loci_bed_fns))
callable_loci_bed_fns[0:2]

def count_symbol(i=1):
    if i%10 == 0:
        return(str((i//10)*10))
    else:
        return('.')

for i, callable_loci_bed_fn in enumerate(callable_loci_bed_fns):
    print('%s' % count_symbol(i), end='', flush=True)
    calc_callable_core(callable_loci_bed_fn)

for callable_loci_bed_fn in callable_loci_bed_fns:
    callable_loci_callable_fn = callable_loci_bed_fn.replace('.bed', '.callable.bed')
    get_ipython().system('grep CALLABLE {callable_loci_bed_fn} > {callable_loci_callable_fn}')

for callable_loci_bed_fn in callable_loci_bed_fns:
    callable_loci_callable_fn = callable_loci_bed_fn.replace('.bed', '.callable.bed')
    get_ipython().system('grep CALLABLE {callable_loci_bed_fn} > {callable_loci_callable_fn}')

get_ipython().system("{bedtools} intersect -a {callable_loci_bed_fn.replace('.bed', '.callable.bed')} -b {core_regions_fn} | {bedtools} genomecov -i - -g {GENOME_FN+'.fai'} | grep -P 'genome\\t1'")
# {bedtools} genomecov -i - -g {GENOME_FN+'.fai'} | grep -P 'genome\t1'

callable_loci_callable_fns = [(callable_loci_bed_fn_format % ox_code).replace('.bed', '.callable.bed') for ox_code in ox_codes]
callable_loci_callable_fns[0:2]

callable_loci_merged_fn = '/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_merged.bed'
for i, callable_loci_callable_fn in enumerate(callable_loci_callable_fns):
    get_ipython().system('cat {callable_loci_callable_fn} >> {callable_loci_merged_fn}')
    print('%d' % (i%10), end='', flush=True)

get_ipython().system("sort -k1,1 -k2,2n {callable_loci_merged_fn} > {callable_loci_merged_fn.replace('.bed', '.sort.bed')}")

get_ipython().system("/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools genomecov -i {callable_loci_merged_fn.replace('.bed', '.sort.bed')} -g {GENOME_FN+'.fai'} -bga > /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_merged_coverage.bed")

bedtools_command = '''{multiIntersectBed} -i {bed_files_list} -empty -g {genome} | cut -f 1-4 > {output_filename}'''.format(
    multiIntersectBed = multiIntersectBed,
    bed_files_list = " ".join(callable_loci_callable_fns),
    genome = GENOME_FN+'.fai',
    output_filename = callable_loci_fn
)
get_ipython().system('{bedtools_command}')

117*13

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

proportion_core_callable = tbl_field_samples_extended.values('core_bases_callable').array()/core_genome_length

20155438 / core_genome_length

proportion_core_callable

for x in [0.98, 0.97, 0.96, 0.95, 0.9, 0.8, 0.5, 0.1, 0.01]:
    print(x, np.sum(proportion_core_callable >= x), np.sum(proportion_core_callable >= x)/2512)

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1, 1, 1)
ax.hist(proportion_core_callable, bins=np.linspace(0.0, 1.0, num=101))
fig.tight_layout()
fig.savefig("%s/proportion_core_callable_histogram.pdf" % plot_dir)

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1, 1, 1)
ax.hist(proportion_core_callable, bins=np.linspace(0.9, 1.0, num=101))
fig.tight_layout()
fig.savefig("%s/proportion_core_callable_histogram_90.pdf" % plot_dir)

ox_codes = tbl_field_samples_extended.selectge('core_bases_callable', core_genome_length*0.95).values('sample').array(dtype='U12')
len(ox_codes)

ox_codes.dtype

2+2

callable_loci_merged_fn = '/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged.bed'
for i, ox_code in enumerate(ox_codes):
    print('%s' % count_symbol(i), end='', flush=True)
    callable_loci_bed_fn = callable_loci_bed_fn_format % ox_code
    get_ipython().system('grep CALLABLE {callable_loci_bed_fn} >> {callable_loci_merged_fn}')

get_ipython().system("sort -k1,1 -k2,2n {callable_loci_merged_fn} > {callable_loci_merged_fn.replace('.bed', '.sort.bed')}")

get_ipython().system("/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools genomecov -i {callable_loci_merged_fn.replace('.bed', '.sort.bed')} -g {GENOME_FN+'.fai'} -bga > /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.bed")

get_ipython().system("/nfs/team112_internal/rp7/opt/bedtools/bedtools2/bin/bedtools genomecov -i {callable_loci_merged_fn.replace('.bed', '.sort.bed')} -g {GENOME_FN+'.fai'} -d > /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.txt")

get_ipython().system('bgzip -f /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.bed')
get_ipython().system('bgzip -f /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.txt')

get_ipython().system('tabix -f -p bed /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.bed.gz')
get_ipython().system('tabix -f -s 1 -b 2 -e 2 /lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.txt.gz')

merged_coverage_fn = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.txt.gz"

accessibility_array = (etl
 .fromtsv(merged_coverage_fn)
 .pushheader(['chrom', 'pos', 'coverage'])
 .convertnumbers()
 .toarray(dtype='a11, i4, i4')
)

merged_coverage_fn = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.txt.gz"
accessibility_array = np.loadtxt(merged_coverage_fn,
                                 dtype={'names': ('chrom', 'pos', 'coverage'), 'formats': ('U11', 'i4', 'i4')})



merged_coverage_fn = "/lustre/scratch109/malaria/rp7/data/pf3k/pilot_5_0/20160525_CallableLoci_bed_release_5/results/callable_loci_gt95_merged_coverage.txt.gz"

accessibility_array = (etl
 .fromtabix(merged_coverage_fn, region = 'Pf3D7_01_v3', header=['chrom', 'pos', 'coverage'])
 .convertnumbers()
 .toarray()
)

print(len(accessibility_array))
print(accessibility_array[0])
accessibility_array

accessibility_array_fn = "%s/accessibility_array.npy" % plot_dir
np.save(accessibility_array_fn, accessibility_array)

del(accessibility_array)
gc.collect()

accessibility_array_fn = "%s/accessibility_array.npy" % plot_dir
accessibility_array = np.load(accessibility_array_fn)

accessibility_array

2+2

chrom.encode('ascii')

accessibility_array[accessibility_array['chrom']==chrom.encode('ascii')]['pos']

(etl
            .fromtsv(REGIONS_FN)
            .pushheader('chrom', 'start', 'end', 'region')
            .convertnumbers()
#              .valuecounts('region').displayall()
        )

accessibility_colors = {
    'Core': 'white',
    'SubtelomericHypervariable': 'red',
    'InternalHypervariable': 'orange',
    'SubtelomericRepeat': 'brown',
    'Centromere': 'black'
#     'InternalHypervariable': '#b20000',
}

def plot_accessibility(bin_size=1000, number_of_samples = 1848):

    fig = plt.figure(figsize=(11.69*1, 8.27*1))
    gs = GridSpec(2*14, 1, height_ratios=([1.0, 1.0])*14)
    gs.update(hspace=0, left=.12, right=.98, top=.98, bottom=.02)

    print('\n', bin_size)
    for i in range(14):
        print(i+1, end=" ")
        chrom = 'Pf3D7_%02d_v3' % (i + 1)
        pos = accessibility_array[accessibility_array['chrom']==chrom.encode('ascii')]['pos']
        coverage = accessibility_array[accessibility_array['chrom']==chrom.encode('ascii')]['coverage']
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
    
    fig.savefig(os.path.join(plot_dir, 'short_read_accesibility_%dbp_windows.png' % bin_size), dpi=150)
    fig.savefig(os.path.join(plot_dir, 'short_read_accesibility_%dbp_windows.pdf' % bin_size))

plot_accessibility()

# for bin_size in [100000, 10000, 1000, 100, 10, 1]:
for bin_size in [100000, 10000, 1000, 1]:
    plot_accessibility(bin_size)

# for bin_size in [100000, 10000, 1000, 100, 10, 1]:
for bin_size in [500, 300]:
    plot_accessibility(bin_size)

# for bin_size in [100000, 10000, 1000, 100, 10, 1]:
for bin_size in [100, 10]:
    plot_accessibility(bin_size)

def plot_accessibility_region(chrom='Pf3D7_10_v3', start=1.4e+6, end=1.44e+6, bin_size=1000, number_of_samples = 1848,
                              tick_distance=5000):

    fig = plt.figure(figsize=(8, 3))

    pos_array = (
        (accessibility_array['chrom']==chrom.encode('ascii')) &
        (accessibility_array['pos']>=start) &
        (accessibility_array['pos']<=end)
    )
    pos = accessibility_array[pos_array]['pos']
    coverage = accessibility_array[pos_array]['coverage']
    min_pos = np.min(pos)
    max_pos = np.max(pos)
    print(min_pos, max_pos)
    if bin_size == 1:
        binned_coverage, bin_centres = coverage, pos
    else:
        binned_coverage, bins, _ = scipy.stats.binned_statistic(pos, coverage, bins=np.arange(min_pos, max_pos, bin_size))
        bin_centres = (bins[:-1]+bins[1:]) / 2
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(bin_centres, binned_coverage/number_of_samples)
    ax.set_xlim(min_pos, max_pos)
#     ax.set_xlim(0, 3300000)
    ax.set_xticks(range(min_pos, max_pos+1, tick_distance))
    ax.set_xticklabels(np.arange(min_pos/1e+6, (max_pos+1)/1e+6, tick_distance/1e+6))
#         for region_chrom, start_pos, end_pos, region_type, region_size in tbl_regions.data():
#             if chrom == region_chrom and region_type != 'Core':
#                 ax.axvspan(start_pos, end_pos, facecolor=accessibility_colors[region_type], alpha=0.1)
    for s in 'left', 'right', 'top':
        ax.spines[s].set_visible(False)
#         ax.set_yticklabels([])
    ax.get_xaxis().tick_bottom()
    ax.set_yticks([])

#     ax.set_ylabel(i+1, rotation='horizontal', horizontalalignment='right', verticalalignment='center')

    ax.set_xlabel('')
    ax.spines['bottom'].set_bounds(min_pos, max_pos)
    
    fig.savefig(os.path.join(plot_dir, 'short_read_accesibility_%dbp_windows_%s_%d_%d.png' % (bin_size, chrom, start, end)), dpi=150)
    fig.savefig(os.path.join(plot_dir, 'short_read_accesibility_%dbp_windows_%s_%d_%d.pdf' % (bin_size, chrom, start, end)))

for bin_size in [1000, 500, 300, 100]:
    plot_accessibility_region(bin_size=bin_size)

plot_accessibility_region(start=1.2e+6, end=1.6e+6, bin_size=300)

# RH2a and RH2b
plot_accessibility_region(chrom='Pf3D7_13_v3', start=1.4e+6, end=1.5e+6, bin_size=300, tick_distance=10000)

# RH2a and RH2b
plot_accessibility_region(chrom='Pf3D7_13_v3', start=1.41e+6, end=1.46e+6, bin_size=300, tick_distance=5000)

# MSP region
plot_accessibility_region(bin_size=300)

# CRT region
plot_accessibility_region(chrom='Pf3D7_07_v3', start=403000, end=406500, bin_size=1, tick_distance=500)

# WG for Thomas
for bin_size in [10000, 1000, 100, 1]:
    plot_accessibility(bin_size)

number_of_samples = 1848

fig = plt.figure(figsize=(11.69*2, 8.27*2))
gs = GridSpec(2*14, 1, height_ratios=([1.0, 0.5])*14)
gs.update(hspace=0, left=.12, right=.98, top=.98, bottom=.02)

for i in range(14):
    print(i, end=" ")
    chrom = 'Pf3D7_%02d_v3' % (i + 1)
    accessibility_array = (etl
        .fromtabix(merged_coverage_fn, region = chrom, header=['chrom', 'pos', 'coverage'])
        .cut(['pos', 'coverage'])
        .convertnumbers()
        .toarray()
    )
    ax = fig.add_subplot(gs[i*2])
    ax.plot(accessibility_array['pos'], accessibility_array['coverage']/number_of_samples)
    ax.set_xlim(0, 3300000)
    ax.set_xticks(range(0, len(core_genome_dict[chrom]), 100000))
    ax.set_xticklabels(range(0, int(len(core_genome_dict[chrom])/1000), 100))
#         for region_chrom, start_pos, end_pos, region_type, region_size in tbl_regions.data():
#             if chrom == region_chrom and region_type != 'Core':
#                 ax.axvspan(start_pos, end_pos, facecolor=accessibility_colors[region_type], alpha=0.1)
#         for s in 'left', 'right':
#             ax.spines[s].set_visible(False)
#             ax.set_yticklabels([])
    ax.set_yticks([])

    ax.set_title(chrom, loc='left')

    ax.set_xlabel('')
    if i < 13:
        ax.set_xticklabels([])
    ax.spines['top'].set_bounds(0, len(core_genome_dict[chrom]))    
    ax.spines['bottom'].set_bounds(0, len(core_genome_dict[chrom]))    

2+2



