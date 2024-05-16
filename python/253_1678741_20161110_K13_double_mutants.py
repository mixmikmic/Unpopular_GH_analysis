get_ipython().run_line_magic('run', '_standard_imports.ipynb')

output_dir = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf 6.0/20161110_K13_double_mutants"
get_ipython().system('mkdir -p {output_dir}')

def create_variants_npy(vcf_fn='/lustre/scratch116/malaria/pfalciparum/output/0/8/b/3/72179/1_gatk_combine_variants_gatk3_v2/SNP_INDEL_Pf3D7_13_v3.combined.vcf.gz',
                        region='Pf3D7_13_v3:1724817-1726997'):
    cache_dir = '%s/vcfnp_cache' % output_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    variants = vcfnp.variants(
        vcf_fn,
        region,
#         fields=['CHROM', 'POS', 'REF', 'ALT',
#                 'AC', 'AN', 'FILTER', 'VQSLOD'],
        dtypes={
            'REF':                      'a200',
            'ALT':                      'a200',
        },
#         arities={
#             'ALT':   1,
#             'AC':    1,
#         },
        flatten_filter=True,
        progress=100,
        verbose=True,
        cache=True,
        cachedir=cache_dir
    )
    return(variants)

def create_calldata_npy(vcf_fn='/lustre/scratch116/malaria/pfalciparum/output/0/8/b/3/72179/1_gatk_combine_variants_gatk3_v2/SNP_INDEL_Pf3D7_13_v3.combined.vcf.gz',
                        region='Pf3D7_13_v3:1724817-1726997'):
    cache_dir = '%s/vcfnp_cache' % output_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    calldata = vcfnp.calldata_2d(
        vcf_fn,
        region,
        fields=['GT', 'AD', 'GQ', 'PGT', 'PID'],
#         dtypes={
#             'AD': 'u2',
#         },
#         arities={
#             'AD': max_alleles,
#         },
        progress=100,
        verbose=True,
        cache=True,
        cachedir=cache_dir
    )
    return(calldata)

variants = create_variants_npy()
calldata = create_calldata_npy()

calldata['GT'].shape

def num_nonref(gt):
    return(np.count_nonzero(np.logical_not(np.in1d(gt, [b'0/0', b'./.']))))
    

def num_nonref_hom(gt):
    return(np.count_nonzero(np.logical_not(np.in1d(gt, [b'0/0', b'./.', b'0/1', b'0/2', b'1/2']))))
    

nonref_per_sample = np.apply_along_axis(num_nonref, 0, calldata['GT'])

len(nonref_per_sample)

variant_set = collections.OrderedDict()
variant_set['all'] = (variants['CHROM'] == b'Pf3D7_13_v3')
variant_set['pass'] = (variants['FILTER_PASS'])
variant_set['non_synonymous'] = (variants['SNPEFF_EFFECT'] == b'NON_SYNONYMO')
variant_set['non_synonymous pass'] = ( (variants['SNPEFF_EFFECT'] == b'NON_SYNONYMO') & (variants['FILTER_PASS']) )
variant_set['BTBPOZ_propeller'] = (variants['POS'] <= 1725953) # from amino acid 349
variant_set['non_synonymous BTBPOZ_propeller'] = ( (variants['SNPEFF_EFFECT'] == b'NON_SYNONYMO') & (variants['POS'] <= 1725953) )
variant_set['non_synonymous pass BTBPOZ_propeller'] = ( (variants['SNPEFF_EFFECT'] == b'NON_SYNONYMO') & (variants['FILTER_PASS']) & (variants['POS'] <= 1725953) )

for set_name in variant_set:
    print(set_name, np.count_nonzero(variant_set[set_name]))
    nonref_per_sample = np.apply_along_axis(num_nonref, 0, calldata['GT'][variant_set[set_name], :])
    print(np.unique(nonref_per_sample, return_counts=True))
    print()

for set_name in variant_set:
    print(set_name, np.count_nonzero(variant_set[set_name]))
    nonref_hom_per_sample = np.apply_along_axis(num_nonref_hom, 0, calldata['GT'][variant_set[set_name], :])
    print(np.unique(nonref_hom_per_sample, return_counts=True))
    print()

nonref_per_sample = np.apply_along_axis(num_nonref, 0, calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :])

vcf_fn = '/lustre/scratch116/malaria/pfalciparum/output/0/8/b/3/72179/1_gatk_combine_variants_gatk3_v2/SNP_INDEL_Pf3D7_13_v3.combined.vcf.gz'
samples = np.array(vcf.Reader(filename=vcf_fn).samples)
samples

print(list(samples[nonref_per_sample>2]))
nonref_per_sample[nonref_per_sample>2]



nonref_per_variant_dodgy = np.apply_along_axis(
    lambda x: np.count_nonzero(np.logical_not(np.in1d(x, [b'0/0', b'./.']))),
    1,
    calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, nonref_per_sample>2]
)
nonref_per_variant_dodgy

nonref_per_variant_all = np.apply_along_axis(
    lambda x: np.count_nonzero(np.logical_not(np.in1d(x, [b'0/0', b'./.']))),
    1,
    calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :]
)
nonref_per_variant_all

print(nonref_per_variant_dodgy[nonref_per_variant_dodgy >= 4])
print(nonref_per_variant_all[nonref_per_variant_dodgy >= 4])

print(list(samples[nonref_per_sample==2]))

calldata['AD'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PC0172-C'][:, 0].transpose()

get_ipython().system('grep PC0172 ~/pf_60_mergelanes.txt')

# This one has 3 K13 mutants, but Olivo has down as Pf only
get_ipython().system('grep PH0714 ~/pf_60_mergelanes.txt')

# This one has 9 K13 mutants, but Olivo has down as Pf/Pm/Po - can we see any Po reads?
get_ipython().system('grep QP0097 ~/pf_60_mergelanes.txt')

# This one has 3 K13 mutants, but Olivo has down as Pf/Pv - can we see any Pv reads?
get_ipython().system('grep PH0914 ~/pf_60_mergelanes.txt')

# This one has 4 K13 mutants, but Olivo has down as Pf/Pv - can we see any Pv reads?
get_ipython().system('grep QC0280 ~/pf_60_mergelanes.txt')

# This one looks like a mixture of two different K13 mutants in proportions ~1:2
calldata['AD'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PD0464-C'][:, 0].transpose()

for sample in samples[nonref_per_sample==2]:
    print(sample)
    print(calldata['AD'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples==sample][:, 0].transpose())
    print()

np.unique(nonref_per_sample, return_counts=True)

calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, nonref_per_sample>2]

calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0190-C'][:, 0]

calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PF0345-C'][:, 0]

calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0271-C'][:, 0]

calldata['GQ'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0271-C'][:, 0]

calldata['GQ'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0271-C'][:, 0][
    calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0271-C'][:, 0] == b'1/1'
]

calldata['AD'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0271-C'][:, 0][
    calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0271-C'][:, 0] == b'1/1'
]

calldata['AD'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0271-C'][:, 0].transpose()

calldata['GQ'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PF0345-C'][:, 0]

calldata['AD'].shape

calldata['GT'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, samples=='PA0190-C'][:, 0]

calldata['AD'][variant_set['non_synonymous pass BTBPOZ_propeller'], :][:, nonref_per_sample>2]

get_ipython().system('grep PA0271 ~/pf_60_mergelanes.txt')
get_ipython().system('grep PF0345 ~/pf_60_mergelanes.txt')
# Then copied these over to macbook to look at in IGV

get_ipython().system('grep PA0271 ~/pf_60_speciator.txt')

get_ipython().system('grep PF0345 ~/pf_60_speciator.txt')

# Looks like PA0271-C isn't P. falciparum
get_ipython().system('cat /lustre/scratch116/malaria/pfalciparum/output/9/7/3/0/48138/1_speciator/6936_2_nonhuman_4.pe.markdup.recal.speciator.tab')

# Looks like PF0345-C is P. falciparum
get_ipython().system('cat /lustre/scratch116/malaria/pfalciparum/output/f/3/b/0/47447/1_speciator/8516_5_29.pe.markdup.recal.speciator.tab')



