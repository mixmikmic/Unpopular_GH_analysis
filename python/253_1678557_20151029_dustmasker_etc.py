get_ipython().run_line_magic('run', '_shared_setup.ipynb')

install_dir = '../opt_4'
REF_GENOME="/lustre/scratch110/malaria/rp7/Pf3k/GATKbuild/Pfalciparum_GeneDB_Aug2015/Pfalciparum.genome.fasta"
regions_fn = '/nfs/users/nfs_r/rp7/src/github/malariagen/pf-crosses/meta/regions-20130225.bed.gz'
# regions_fn = '/Users/rpearson/src/github/malariagen/pf-crosses/meta/regions-20130225.bed.gz'
ref_gff = "%s/snpeff/snpEff/data/Pfalciparum_GeneDB_Aug2015/genes.gff" % install_dir
ref_cds_gff = REF_GENOME.replace('.fasta', '.CDS.gff')

get_ipython().system("head -n -34 {ref_gff} | grep -P '\\tCDS\\t' > {ref_cds_gff}")

# !wget ftp://ftp.ncbi.nlm.nih.gov/pub/agarwala/dustmasker/dustmasker -O {install_dir}/dustmasker
# !chmod a+x {install_dir}/dustmasker

get_ipython().system('wget ftp://ftp.ncbi.nlm.nih.gov/pub/agarwala/windowmasker/windowmasker -O {install_dir}/windowmasker')
get_ipython().system('chmod a+x {install_dir}/windowmasker')

current_dir = get_ipython().getoutput('pwd')
current_dir = current_dir[0]
get_ipython().system('wget http://cbrc3.cbrc.jp/~martin/tantan/tantan-13.zip -O {install_dir}/tantan-13.zip')
get_ipython().run_line_magic('cd', '{install_dir}')
get_ipython().system('unzip tantan-13.zip')
get_ipython().run_line_magic('cd', 'tantan-13')
get_ipython().system('make')
get_ipython().run_line_magic('cd', '{current_dir}')

ref_dict=SeqIO.to_dict(SeqIO.parse(open(REF_GENOME), "fasta"))
chromosome_lengths = [len(ref_dict[chrom]) for chrom in ref_dict]
tbl_chromosomes=(etl.wrap(zip(ref_dict.keys(), chromosome_lengths))
    .pushheader(['chrom', 'stop'])
    .addfield('start', 0)
    .cut(['chrom', 'start', 'stop'])
    .sort('chrom')
)
tbl_chromosomes

tbl_regions = (etl
    .fromtsv(regions_fn)
    .pushheader(['chrom', 'start', 'stop', 'region'])
    .convertnumbers()
)
tbl_regions.display(10)

iscore_array = collections.OrderedDict()
for chromosomes_row in tbl_chromosomes.data():
    chrom=chromosomes_row[0]
    iscore_array[chrom] = np.zeros(chromosomes_row[2], dtype=bool)
    for regions_row in tbl_regions.selecteq('chrom', chrom).selecteq('region', 'Core').data():
        iscore_array[chrom][regions_row[1]:regions_row[2]] = True

tbl_ref_cds_gff = (
    etl.fromgff3(ref_cds_gff)
    .select(lambda rec: rec['end'] > rec['start'])
    .unpackdict('attributes')
    .select(lambda rec: rec['Parent'].endswith('1')) # Think there are alternate splicings for some genes, here just using first
    .distinct(['seqid', 'start'])
)

tbl_coding_regions = (tbl_ref_cds_gff
    .cut(['seqid', 'start', 'end'])
    .rename('end', 'stop')
    .rename('seqid', 'chrom')
    .convert('start', lambda val: val-1)
)
tbl_coding_regions                   

iscoding_array = collections.OrderedDict()
for chromosomes_row in tbl_chromosomes.data():
    chrom=chromosomes_row[0]
    iscoding_array[chrom] = np.zeros(chromosomes_row[2], dtype=bool)
    for coding_regions_row in tbl_coding_regions.selecteq('chrom', chrom).data():
        iscoding_array[chrom][coding_regions_row[1]:coding_regions_row[2]] = True

def which_lower(string):
    return np.array([str.islower(x) for x in string])
which_lower('abCDeF') 
# np.array([str.islower(x) for x in 'abCDeF'])

def find_regions(masked_pos, number_of_regions=3):
    masked_regions = list()
    start = masked_pos[0]
    stop = start
    region_number = 1
    for pos in masked_pos:
        if pos > (stop + 1):
            masked_regions.append([start, stop])
            start = stop = pos
            region_number = region_number + 1
            if region_number > number_of_regions:
                break
        else:
            stop = pos
    return(masked_regions)

def summarise_masking(
    classification_array,
    masking_description = "Dust level 20",
    number_of_regions = 10,
    max_sequence_length = 60
):
    number_core_coding_masked = np.count_nonzero(classification_array['Core coding masked'])
    number_core_coding_unmasked = np.count_nonzero(classification_array['Core coding unmasked'])
    number_core_noncoding_masked = np.count_nonzero(classification_array['Core noncoding masked'])
    number_core_noncoding_unmasked = np.count_nonzero(classification_array['Core noncoding unmasked'])
    proportion_core_coding_masked = number_core_coding_masked / (number_core_coding_masked + number_core_coding_unmasked)
    proportion_core_noncoding_masked = number_core_noncoding_masked / (number_core_noncoding_masked + number_core_noncoding_unmasked)
    print("%s: %4.1f%% coding and %4.1f%% non-coding masked" % (
            masking_description,
            proportion_core_coding_masked*100,
            proportion_core_noncoding_masked*100
        )
    )
    coding_masked_pos = np.where(classification_array['Core coding masked'])[0]
    noncoding_masked_pos = np.where(classification_array['Core noncoding masked'])[0]
    coding_masked_regions = find_regions(coding_masked_pos, number_of_regions)
    noncoding_masked_regions = find_regions(noncoding_masked_pos, number_of_regions)    
    
    print("    First %d Pf3D7_01_v3 coding sequences masked:" % number_of_regions)
    for region in coding_masked_regions:
        if region[1] - region[0] > max_sequence_length:
            masked_sequence = "%s[...]" % ref_dict['Pf3D7_01_v3'].seq[region[0]:region[1]][0:max_sequence_length]
        else:
            masked_sequence = ref_dict['Pf3D7_01_v3'].seq[region[0]:region[1]]
        print("        %d - %d: %s" % (
                region[0],
                region[1],
                masked_sequence
            )
        )
    print("    First %d Pf3D7_01_v3 non-coding sequences masked:" % number_of_regions)
    for region in noncoding_masked_regions:
        if region[1] - region[0] > max_sequence_length:
            masked_sequence = "%s[...]" % ref_dict['Pf3D7_01_v3'].seq[region[0]:region[1]][0:max_sequence_length]
        else:
            masked_sequence = ref_dict['Pf3D7_01_v3'].seq[region[0]:region[1]]
        print("        %d - %d: %s" % (
                region[0],
                region[1],
                masked_sequence
            )
        )
    print()
    

def evaluate_dust_threshold(
    dust_level=20,
    verbose=False
):
    masked_genome_fn = "%s.dustmasker.%d.fasta" % (REF_GENOME.replace('.fasta', ''), dust_level)
    
    if verbose:
        print("Running dustmasker %d" % dust_level)
    get_ipython().system('{install_dir}/dustmasker     -in {REF_GENOME}     -outfmt fasta     -out {masked_genome_fn}     -level {dust_level}')

    if verbose:
        print("Reading in fasta %d" % dust_level)
    masked_ref_dict=SeqIO.to_dict(SeqIO.parse(open(masked_genome_fn), "fasta"))

    if verbose:
        print("Creating mask array %d" % dust_level)
    ismasked_array = collections.OrderedDict()
    classification_array = collections.OrderedDict()
    
    genome_length = sum([len(ref_dict[chrom]) for chrom in ref_dict])
    for region_type in [
        'Core coding unmasked',
        'Core coding masked',
        'Core noncoding unmasked',
        'Core noncoding masked',
        'Noncore coding unmasked',
        'Noncore coding masked',
        'Noncore noncoding unmasked',
        'Noncore noncoding masked',
    ]:
        classification_array[region_type] = np.zeros(genome_length, dtype=bool)
        
    offset=0
    for chromosomes_row in tbl_chromosomes.data():
        chrom=chromosomes_row[0]
        masked_ref_dict_chrom = "lcl|%s" % chrom
        if verbose:
            print(chrom)
        chrom_length=chromosomes_row[2]
        ismasked_array[chrom] = which_lower(masked_ref_dict[masked_ref_dict_chrom].seq)
        classification_array['Core coding unmasked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & iscoding_array[chrom] & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Core coding masked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & iscoding_array[chrom] & ismasked_array[chrom]
        )
        classification_array['Core noncoding unmasked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & np.logical_not(iscoding_array[chrom]) & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Core noncoding masked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & np.logical_not(iscoding_array[chrom]) & ismasked_array[chrom]
        )
        classification_array['Noncore coding unmasked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & iscoding_array[chrom] & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Noncore coding masked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & iscoding_array[chrom] & ismasked_array[chrom]
        )
        classification_array['Noncore noncoding unmasked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & np.logical_not(iscoding_array[chrom]) & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Noncore noncoding masked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & np.logical_not(iscoding_array[chrom]) & ismasked_array[chrom]
        )
        offset = offset + chrom_length

    summarise_masking(classification_array, "Dust level %d" % dust_level)
                      
#     number_core_coding_masked = np.count_nonzero(classification_array['Core coding masked'])
#     number_core_coding_unmasked = np.count_nonzero(classification_array['Core coding unmasked'])
#     number_core_noncoding_masked = np.count_nonzero(classification_array['Core noncoding masked'])
#     number_core_noncoding_unmasked = np.count_nonzero(classification_array['Core noncoding unmasked'])
#     proportion_core_coding_masked = number_core_coding_masked / (number_core_coding_masked + number_core_coding_unmasked)
#     proportion_core_noncoding_masked = number_core_noncoding_masked / (number_core_noncoding_masked + number_core_noncoding_unmasked)
#     print("dustmasker dust_level=%d: %4.1f%% coding and %4.1f%% non-coding masked" % (
#             dust_level,
#             proportion_core_coding_masked*100,
#             proportion_core_noncoding_masked*100,
# #             ''.join(np.array(ref_dict['Pf3D7_01_v3'].seq)[classification_array['Core coding masked'][0:640851]])[0:60]
#        )
#     )
#     non_coding_masked_pos = np.where(classification_array['Core coding masked'])[0]
#     for masked_coding_region in find_regions(non_coding_masked_pos):
#         print("\t%d - %d: %s" % (
#                 masked_coding_region[0],
#                 masked_coding_region[1],
#                 ref_dict['Pf3D7_01_v3'].seq[masked_coding_region[0]:masked_coding_region[1]]
#             )
#         )

    return(classification_array, masked_ref_dict, ismasked_array)

def evaluate_windowmasker(
    check_dup='true',
    use_dust='false',
    verbose=False
):
    ustat_fn = "%s.windowmasker.%s.%s.ustat" % (REF_GENOME.replace('.fasta', ''), check_dup, use_dust)
    masked_genome_fn = "%s.windowmasker.%s.%s.fasta" % (REF_GENOME.replace('.fasta', ''), check_dup, use_dust)
    
    if verbose:
        print("Running dustmasker check_dup=%s use_dust=%s" % (check_dup, use_dust))
    get_ipython().system('{install_dir}/windowmasker -mk_counts     -in {REF_GENOME}     -checkdup {check_dup}     -out {ustat_fn}')

    get_ipython().system('{install_dir}/windowmasker     -ustat {ustat_fn}     -in {REF_GENOME}     -outfmt fasta     -out {masked_genome_fn}     -dust {use_dust} ')
    if verbose:
        print("Reading in fasta check_dup=%s use_dust=%s" % (check_dup, use_dust))
    masked_ref_dict=SeqIO.to_dict(SeqIO.parse(open(masked_genome_fn), "fasta"))

    if verbose:
        print("Creating mask array check_dup=%s use_dust=%s" % (check_dup, use_dust))
    ismasked_array = collections.OrderedDict()
    classification_array = collections.OrderedDict()
    
    genome_length = sum([len(ref_dict[chrom]) for chrom in ref_dict])
    for region_type in [
        'Core coding unmasked',
        'Core coding masked',
        'Core noncoding unmasked',
        'Core noncoding masked',
        'Noncore coding unmasked',
        'Noncore coding masked',
        'Noncore noncoding unmasked',
        'Noncore noncoding masked',
    ]:
        classification_array[region_type] = np.zeros(genome_length, dtype=bool)
        
    offset=0
    for chromosomes_row in tbl_chromosomes.data():
        chrom=chromosomes_row[0]
        if verbose:
            print(chrom)
        chrom_length=chromosomes_row[2]
        ismasked_array[chrom] = which_lower(masked_ref_dict[chrom].seq)
        classification_array['Core coding unmasked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & iscoding_array[chrom] & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Core coding masked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & iscoding_array[chrom] & ismasked_array[chrom]
        )
        classification_array['Core noncoding unmasked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & np.logical_not(iscoding_array[chrom]) & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Core noncoding masked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & np.logical_not(iscoding_array[chrom]) & ismasked_array[chrom]
        )
        classification_array['Noncore coding unmasked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & iscoding_array[chrom] & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Noncore coding masked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & iscoding_array[chrom] & ismasked_array[chrom]
        )
        classification_array['Noncore noncoding unmasked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & np.logical_not(iscoding_array[chrom]) & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Noncore noncoding masked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & np.logical_not(iscoding_array[chrom]) & ismasked_array[chrom]
        )
        offset = offset + chrom_length

    summarise_masking(classification_array, "windowmasker check_dup=%s use_dust=%s" % (check_dup, use_dust))

#     number_core_coding_masked = np.count_nonzero(classification_array['Core coding masked'])
#     number_core_coding_unmasked = np.count_nonzero(classification_array['Core coding unmasked'])
#     number_core_noncoding_masked = np.count_nonzero(classification_array['Core noncoding masked'])
#     number_core_noncoding_unmasked = np.count_nonzero(classification_array['Core noncoding unmasked'])
#     proportion_core_coding_masked = number_core_coding_masked / (number_core_coding_masked + number_core_coding_unmasked)
#     proportion_core_noncoding_masked = number_core_noncoding_masked / (number_core_noncoding_masked + number_core_noncoding_unmasked)
#     print("windowmasker check_dup=%s use_dust=%s: %4.1f%% coding and %4.1f%% non-coding masked\n\t%s" % (
#             check_dup,
#             use_dust,
#             proportion_core_coding_masked*100,
#             proportion_core_noncoding_masked*100,
#             ''.join(np.array(ref_dict['Pf3D7_01_v3'].seq)[classification_array['Core coding masked'][0:640851]])[0:60]
#         )
#     )
        
    return(classification_array, masked_ref_dict, ismasked_array)

def evaluate_tantan(
    r=0.005,
    m=None,
    verbose=False
):
    masked_genome_fn = "%s.tantan.%s.%s.fasta" % (REF_GENOME.replace('.fasta', ''), r, m)
    
    if verbose:
        print("Running tantan r=%s m=%s" % (r, m))
    if m is None:
        get_ipython().system('{install_dir}/tantan-13/src/tantan -r {r} {REF_GENOME} > {masked_genome_fn}')
    elif m == 'atMask.mat':
        get_ipython().system('{install_dir}/tantan-13/src/tantan -r {r} -m {install_dir}/tantan-13/test/atMask.mat {REF_GENOME} >             {masked_genome_fn}')
    else:
        stop("Unknown option m=%s" % m)

    if verbose:
        print("Reading in fasta r=%s m=%s" % (r, m))
    masked_ref_dict=SeqIO.to_dict(SeqIO.parse(open(masked_genome_fn), "fasta"))

    if verbose:
        print("Creating mask array r=%s m=%s" % (r, m))
    ismasked_array = collections.OrderedDict()
    classification_array = collections.OrderedDict()
    
    genome_length = sum([len(ref_dict[chrom]) for chrom in ref_dict])
    for region_type in [
        'Core coding unmasked',
        'Core coding masked',
        'Core noncoding unmasked',
        'Core noncoding masked',
        'Noncore coding unmasked',
        'Noncore coding masked',
        'Noncore noncoding unmasked',
        'Noncore noncoding masked',
    ]:
        classification_array[region_type] = np.zeros(genome_length, dtype=bool)
        
    offset=0
    for chromosomes_row in tbl_chromosomes.data():
        chrom=chromosomes_row[0]
        if verbose:
            print(chrom)
        chrom_length=chromosomes_row[2]
        ismasked_array[chrom] = which_lower(masked_ref_dict[chrom].seq)
        classification_array['Core coding unmasked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & iscoding_array[chrom] & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Core coding masked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & iscoding_array[chrom] & ismasked_array[chrom]
        )
        classification_array['Core noncoding unmasked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & np.logical_not(iscoding_array[chrom]) & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Core noncoding masked'][offset:(offset+chrom_length)] = (
            iscore_array[chrom] & np.logical_not(iscoding_array[chrom]) & ismasked_array[chrom]
        )
        classification_array['Noncore coding unmasked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & iscoding_array[chrom] & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Noncore coding masked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & iscoding_array[chrom] & ismasked_array[chrom]
        )
        classification_array['Noncore noncoding unmasked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & np.logical_not(iscoding_array[chrom]) & np.logical_not(ismasked_array[chrom])
        )
        classification_array['Noncore noncoding masked'][offset:(offset+chrom_length)] = (
            np.logical_not(iscore_array[chrom]) & np.logical_not(iscoding_array[chrom]) & ismasked_array[chrom]
        )
        offset = offset + chrom_length

    summarise_masking(classification_array, "tantan r=%s m=%s" % (r, m))

#     number_core_coding_masked = np.count_nonzero(classification_array['Core coding masked'])
#     number_core_coding_unmasked = np.count_nonzero(classification_array['Core coding unmasked'])
#     number_core_noncoding_masked = np.count_nonzero(classification_array['Core noncoding masked'])
#     number_core_noncoding_unmasked = np.count_nonzero(classification_array['Core noncoding unmasked'])
#     proportion_core_coding_masked = number_core_coding_masked / (number_core_coding_masked + number_core_coding_unmasked)
#     proportion_core_noncoding_masked = number_core_noncoding_masked / (number_core_noncoding_masked + number_core_noncoding_unmasked)
#     print("tantan r=%s m=%s: %4.1f%% coding and %4.1f%% non-coding masked\n\t%s" % (
#             r,
#             m,
#             proportion_core_coding_masked*100,
#             proportion_core_noncoding_masked*100,
#             ''.join(np.array(ref_dict['Pf3D7_01_v3'].seq)[classification_array['Core coding masked'][0:640851]])[0:60]
#         )
#     )
        
    return(classification_array, masked_ref_dict, ismasked_array)

dustmasker_classification_arrays = collections.OrderedDict()
for dust_level in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
    dustmasker_classification_arrays[str(dust_level)] = evaluate_dust_threshold(dust_level, verbose=False)

windowmasker_classification_arrays = collections.OrderedDict()
for check_dup in ['true', 'false']:
    windowmasker_classification_arrays[check_dup] = collections.OrderedDict()
    for use_dust in ['true', 'false']:
        windowmasker_classification_arrays[check_dup][use_dust] = evaluate_windowmasker(check_dup, use_dust, verbose=False)

tantan_classification_arrays = collections.OrderedDict()
for m in ['atMask.mat', None]:
    tantan_classification_arrays[str(m)] = collections.OrderedDict()
#     for r in [0.000000000001, 0.000000001, 0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]:
    for r in [0.000000000001, 0.000000001, 0.000001, 0.001, 0.1]:
        tantan_classification_arrays[str(m)][str(r)] = evaluate_tantan(r, m, verbose=False)

str(ref_dict['Pf3D7_01_v3'].seq[100362:100536])

str(ref_dict['Pf3D7_01_v3'].seq[101707:101957])

for dust_level in [20, 30, 40, 50, 60, 70]:
    summarise_masking(
        dustmasker_classification_arrays[str(dust_level)][0],
        "Dust level %d" % dust_level
    )

for check_dup in ['true', 'false']:
    for use_dust in ['true', 'false']:
        summarise_masking(
            windowmasker_classification_arrays[check_dup][use_dust][0],
            "windowmasker check_dup=%s use_dust=%s" % (check_dup, use_dust)
        )

# for r in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]:
for r in [0.000000000001, 0.000000001, 0.000001, 0.001, 0.1]:
    for m in [None, 'atMask.mat']:
        summarise_masking(
            tantan_classification_arrays[str(m)][str(r)][0],
            "tantan r=%s m=%s" % (r, m)
        )
       





import pickle
dustmasker_classification_arrays_fn = REF_GENOME.replace('.fasta', 'dustmasker_classification_arrays.p')
pickle.dump(dustmasker_classification_arrays, open(dustmasker_classification_arrays_fn, "wb"))







