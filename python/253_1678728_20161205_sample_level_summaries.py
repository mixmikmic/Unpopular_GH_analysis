get_ipython().run_line_magic('run', '_standard_imports.ipynb')
get_ipython().run_line_magic('run', '_plotting_setup.ipynb')

output_dir = '/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161205_sample_level_summaries'
get_ipython().system('mkdir -p {output_dir}/sample_summaries/Pf60')
get_ipython().system('mkdir -p {output_dir}/sample_summaries/Pv30')
get_ipython().system('mkdir -p {output_dir}/scripts')
get_ipython().system('mkdir -p {output_dir}/log')

GENOME_FN = collections.OrderedDict()
genome_fn = collections.OrderedDict()
genome = collections.OrderedDict()
GENOME_FN['Pf60'] = "/lustre/scratch116/malaria/pfalciparum/resources/Pfalciparum.genome.fasta"
GENOME_FN['Pv30'] = "/lustre/scratch109/malaria/pvivax/resources/gatk/PvivaxP01.genome.fasta"
genome_fn['Pf60'] = "%s/Pfalciparum.genome.fasta" % output_dir
genome_fn['Pv30'] = "%s/PvivaxP01.genome.fasta" % output_dir

run_create_sample_summary_job_fn = "%s/scripts/run_create_sample_summary_job.sh" % output_dir
submit_create_sample_summary_jobs_fn = "%s/scripts/submit_create_sample_summary_jobs.sh" % output_dir


# sites_annotation_pf60_fn = '/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161125_Pf60_final_vcfs/vcf/SNP_INDEL_WG.combined.filtered.annotation.vcf.gz'
hdf_fn = collections.OrderedDict()
hdf_fn['Pf60'] = '/lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161128_HDF5_build/hdf5/Pf_60_npy_PID_a12.h5'
hdf_fn['Pv30'] = '/lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161201_Pv_30_HDF5_build/hdf5/Pv_30.h5'

for release in GENOME_FN:
    get_ipython().system('cp {GENOME_FN[release]} {genome_fn[release]}')
    genome[release] = pyfasta.Fasta(genome_fn[release])
    print(sorted(genome[release].keys())[0])

hdf = collections.OrderedDict()
for release in hdf_fn:
    hdf[release] = h5py.File(hdf_fn[release], 'r')
    print(release, len(hdf[release]['samples']))
    

get_ipython().run_cell_magic('time', '', 'import pickle\nimport allel\ncalldata_subset = collections.OrderedDict()\nfor release in hdf_fn:\n    calldata_subset[release] = collections.OrderedDict()\n    for variable in [\'genotype\', \'GQ\', \'DP\', \'PGT\']:\n        calldata_subset[release][variable] = collections.OrderedDict()\n        calldata = allel.GenotypeChunkedArray(hdf[release][\'calldata\'][variable])\n        \n        calldata_subset_fn = "%s/calldata_subset_%s_%s_first.p" % (output_dir, release, variable)\n        if os.path.exists(calldata_subset_fn):\n            print(\'loading\', release, variable, \'first\')\n            calldata_subset[release][variable][\'first\'] = pickle.load(open(calldata_subset_fn, \'rb\'))\n        else:\n            print(\'creating\', release, variable, \'first\')\n            calldata_subset[release][variable][\'first\'] = calldata.subset(\n                (hdf[release][\'variants\'][\'CHROM\'][:] == sorted(genome[release].keys())[0].encode(\'ascii\'))\n            )\n            \n        calldata_subset_fn = "%s/calldata_subset_%s_%s_first_pass.p" % (output_dir, release, variable)\n        if os.path.exists(calldata_subset_fn):\n            print(\'loading\', release, variable, \'first_pass\')\n            calldata_subset[release][variable][\'first_pass\'] = pickle.load(open(calldata_subset_fn, \'rb\'))\n        else:\n            print(\'creating\', release, variable, \'first_pass\')\n            calldata_subset[release][variable][\'first_pass\'] = calldata.subset(\n                (hdf[release][\'variants\'][\'FILTER_PASS\'][:]) &\n                (hdf[release][\'variants\'][\'CHROM\'][:] == sorted(genome[release].keys())[0].encode(\'ascii\'))\n            )\n            \n#         calldata_subset_fn = "%s/calldata_subset_%s_%s_pass.p" % (output_dir, release, variable)\n#         if os.path.exists(calldata_subset_fn):\n#             print(\'loading\', release, variable, \'pass\')\n#             calldata_subset[release][variable][\'pass\'] = pickle.load(open(calldata_subset_fn, \'rb\'))\n#         else:\n#             print(\'creating\', release, variable, \'pass\')\n#             calldata_subset[release][variable][\'pass\'] = calldata.subset(hdf[release][\'variants\'][\'FILTER_PASS\'][:])\n            \n#         calldata_subset_fn = "%s/calldata_subset_%s_%s_all.p" % (output_dir, release, variable)\n#         if os.path.exists(calldata_subset_fn):\n#             print(\'loading\', release, variable, \'all\')\n#             calldata_subset[release][variable][\'all\'] = pickle.load(open(calldata_subset_fn, \'rb\'))\n#         else:\n#             print(\'creating\', release, variable, \'all\')\n#             calldata_subset[release][variable][\'all\'] = calldata.subset()\n            \n#             print(release, \'pass\')\n#             calldata_subset[release][variable][\'pass\'] = calldata.subset(hdf[release][\'variants\'][\'FILTER_PASS\'][:])\n#             print(release, \'all\')\n#             calldata_subset[release][variable][\'all\'] = calldata.subset()\n        pickle.dump(genotypes_subset, open(genotypes_subset_fn, "wb"))')

# %%time
# import pickle
# genotypes_subset_fn = "%s/genotypes_subset.p" % output_dir
# if os.path.exists(genotypes_subset_fn):
#     genotypes_subset = pickle.load(open(genotypes_subset_fn, "rb"))
# else:
#     genotypes = collections.OrderedDict()
#     genotypes_subset = collections.OrderedDict()
#     import allel
#     for release in hdf_fn:
#     # for release in ['Pv30']:
#         genotypes[release] = allel.GenotypeChunkedArray(hdf[release]['calldata']['genotype'])
#         genotypes_subset[release] = collections.OrderedDict()
#         print(release, 'first')
#         genotypes_subset[release]['first'] = genotypes[release].subset(
#             (hdf[release]['variants']['FILTER_PASS'][:]) &
#             (hdf[release]['variants']['CHROM'][:] == sorted(genome[release].keys())[0].encode('ascii'))
#         )
#         print(release, 'pass')
#         genotypes_subset[release]['pass'] = genotypes[release].subset(hdf[release]['variants']['FILTER_PASS'][:])
#         print(release, 'all')
#         genotypes_subset[release]['all'] = genotypes[release].subset()
#     pickle.dump(genotypes_subset, open(genotypes_subset_fn, "wb"))

get_ipython().run_cell_magic('time', '', 'import pickle\nGQ_subset_fn = "%s/GQ_subset.p" % output_dir\nif os.path.exists(GQ_subset_fn):\n    genotypes_subset = pickle.load(GQ_subset_fn)\nelse:\n    GQ = collections.OrderedDict()\n    GQ_subset = collections.OrderedDict()\n    for release in hdf_fn:\n        GQ[release] = allel.GenotypeChunkedArray(hdf[release][\'calldata\'][\'GQ\'])\n        GQ_subset[release] = collections.OrderedDict()\n        print(release, \'first\')\n        GQ_subset[release][\'first\'] = GQ[release].subset(\n            (hdf[release][\'variants\'][\'FILTER_PASS\'][:]) &\n            (hdf[release][\'variants\'][\'CHROM\'][:] == sorted(genome[release].keys())[0].encode(\'ascii\'))\n        )\n        print(release, \'pass\')\n        GQ_subset[release][\'pass\'] = GQ[release].subset(hdf[release][\'variants\'][\'FILTER_PASS\'][:])\n        print(release, \'all\')\n        GQ_subset[release][\'all\'] = GQ[release].subset()\n    pickle.dump(GQ_subset, open(GQ_subset_fn, "wb"))')

pickle.dump(genotypes_subset, open(genotypes_subset_fn, "wb"))

temp = (
    (hdf[release]['variants']['FILTER_PASS'][:]) &
    (hdf[release]['variants']['CHROM'][:] == sorted(genome[release].keys())[0].encode('ascii'))
)
pd.value_counts(temp)

sorted(genome[release].keys())[0].encode('ascii')

pd.value_counts(hdf[release]['variants']['CHROM'][:])

hdf[release]['samples'][:]

genotypes_subset['Pf60']

genotypes_subset['Pv30']

import allel
def create_sample_summary(hdf5_fn=hdf_fn['Pf60'], index=0, output_filestem="%s/sample_summaries/Pf60" % output_dir):
    results = collections.OrderedDict()
    
    hdf = h5py.File(hdf5_fn, 'r')
    samples = hdf['samples'][:]
    results['sample_id'] = samples[index].decode('ascii')
    output_fn = "%s/%s.txt" % (output_filestem, results['sample_id'])
    fo = open(output_fn, 'w')
    print(0)
    is_pass = hdf['variants']['FILTER_PASS'][:]
    
#     genotypes = allel.GenotypeChunkedArray(hdf['calldata']['genotype'])
    print(1)
    genotypes = allel.GenotypeArray(hdf['calldata']['genotype'][:, [index], :])
    print(2)
    genotypes_pass = genotypes[is_pass]
    
    print(3)
    svlen = hdf['variants']['svlen'][:]
    svlen1 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 0] - 1]
    svlen1[np.in1d(genotypes[:, 0, 0], [-1, 0])] = 0 # Ref and missing are considered non-SV
    svlen2 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 1] - 1]
    svlen2[np.in1d(genotypes[:, 0, 1], [-1, 0])] = 0 # Ref and missing are considered non-SV
    
    print(4)
    is_snp = (hdf['variants']['VARIANT_TYPE'][:][is_pass] == b'SNP')
    is_bi = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'BI')
    is_sd = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'SD')
    is_mu = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'MU')
    is_ins = ((svlen1 > 0) | (svlen2 > 0))[is_pass]
    is_del = ((svlen1 < 0) | (svlen2 < 0))[is_pass]
    is_ins_del_het = (((svlen1 > 0) & (svlen2 < 0)) | ((svlen1 < 0) & (svlen2 > 0)))[is_pass] # These are hets where one allele is insertion and the other allele is deletion (these are probably rare)
    
    print(5)
    results['num_variants']      = genotypes.shape[0]
    results['num_pass_variants'] = np.count_nonzero(is_pass)
    results['num_missing']       = genotypes.count_missing(axis=0)[0]
    results['num_pass_missing']  = genotypes_pass.count_missing(axis=0)[0]
    results['num_called']        = (results['num_variants'] - results['num_missing'])
    results['num_pass_called']   = (results['num_pass_variants'] - results['num_pass_missing'])
    print(6)
    results['num_het']           = genotypes.count_het(axis=0)[0]
    results['num_pass_het']      = genotypes_pass.count_het(axis=0)[0]
    results['num_hom_alt']       = genotypes.count_hom_alt(axis=0)[0]
    results['num_pass_hom_alt']  = genotypes_pass.count_hom_alt(axis=0)[0]
    print(7)
    results['num_snp_hom_ref']   = genotypes_pass.subset(is_snp).count_hom_ref(axis=0)[0]
    results['num_snp_het']       = genotypes_pass.subset(is_snp).count_het(axis=0)[0]
    results['num_snp_hom_alt']   = genotypes_pass.subset(is_snp).count_hom_alt(axis=0)[0]
    results['num_indel_hom_ref'] = genotypes_pass.subset(~is_snp).count_hom_ref(axis=0)[0]
    results['num_indel_het']     = genotypes_pass.subset(~is_snp).count_het(axis=0)[0]
    results['num_indel_hom_alt'] = genotypes_pass.subset(~is_snp).count_hom_alt(axis=0)[0]
    print(8)    
    results['num_ins_hom_ref']   = genotypes_pass.subset(is_ins).count_hom_ref(axis=0)[0]
    results['num_ins_het']       = genotypes_pass.subset(is_ins).count_het(axis=0)[0]
    results['num_ins']           = (results['num_ins_hom_ref'] + results['num_ins_het'])
    results['num_del_hom_ref']   = genotypes_pass.subset(is_del).count_hom_ref(axis=0)[0]
    results['num_del_het']       = genotypes_pass.subset(is_del).count_het(axis=0)[0]
    results['num_del']           = (results['num_del_hom_ref'] + results['num_del_het'])
    
    print(9)
    results['pc_pass']           = results['num_pass_called'] / results['num_called']
    results['pc_missing']        = results['num_missing'] / results['num_variants']
    results['pc_pass_missing']   = results['num_pass_missing'] / results['num_pass_variants']
    results['pc_het']            = results['num_het'] / results['num_called']
    results['pc_pass_het']       = results['num_pass_het'] / results['num_pass_called']
    results['pc_hom_alt']        = results['num_hom_alt'] / results['num_called']
    results['pc_pass_hom_alt']   = results['num_pass_hom_alt'] / results['num_pass_called']
    results['pc_snp']            = (results['num_snp_het'] + results['num_snp_hom_alt']) / (results['num_snp_het'] + results['num_snp_hom_alt'] + results['num_indel_het'] + results['num_indel_hom_alt'])
    results['pc_ins']            = (results['num_ins'] / (results['num_ins'] + results['num_del']))

    print(10)
    
    print('\t'.join([str(x) for x in list(results.keys())]), file=fo)
    print('\t'.join([str(x) for x in list(results.values())]), file=fo)
    fo.close()
    
    df_sample_summary = pd.DataFrame(
            {
                'Sample': pd.Series(results['sample_id']),
                'Variants called': pd.Series(results['num_called']),
                'Variants missing': pd.Series(results['num_missing']),
                'Proportion missing': pd.Series(results['pc_missing']),
                'Proportion pass missing': pd.Series(results['pc_pass_missing']),
                'Proportion heterozygous': pd.Series(results['pc_het']),
                'Proportion pass heterozygous': pd.Series(results['pc_pass_het']),
                'Proportion homozygous alternative': pd.Series(results['pc_hom_alt']),
                'Proportion pass homozygous alternative': pd.Series(results['pc_pass_hom_alt']),
                'Proportion variants SNPs': pd.Series(results['pc_snp']),
                'Proportion indels insertions': pd.Series(results['pc_ins']),
            }
        )  
    return(df_sample_summary, results)

hdf_fn['Pf60']

import allel
def create_sample_summary(index=0, hdf5_fn='/lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161128_HDF5_build/hdf5/Pf_60_npy_PID_a12.h5'):
    results = collections.OrderedDict()
    
    hdf = h5py.File(hdf5_fn, 'r')
    samples = hdf['samples'][:]
    results['sample_id'] = samples[index].decode('ascii')
#     output_fn = "%s/%s.txt" % (output_filestem, results['sample_id'])
#     fo = open(output_fn, 'w')
    is_pass = hdf['variants']['FILTER_PASS'][:]
    
    genotypes = allel.GenotypeArray(hdf['calldata']['genotype'][:, [index], :])
    genotypes_pass = genotypes[is_pass]
    
    svlen = hdf['variants']['svlen'][:]
    svlen1 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 0] - 1]
    svlen1[np.in1d(genotypes[:, 0, 0], [-1, 0])] = 0 # Ref and missing are considered non-SV
    svlen2 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 1] - 1]
    svlen2[np.in1d(genotypes[:, 0, 1], [-1, 0])] = 0 # Ref and missing are considered non-SV
    
    ac = hdf['variants']['AC'][:]
    ac1 = ac[np.arange(ac.shape[0]), genotypes[:, 0, 0] - 1]
    ac1[np.in1d(genotypes[:, 0, 0], [-1, 0])] = 0 # Ref and missing are considered non-singleton
    ac2 = ac[np.arange(ac.shape[0]), genotypes[:, 0, 1] - 1]
    ac2[np.in1d(genotypes[:, 0, 1], [-1, 0])] = 0 # Ref and missing are considered non-singleton
    
    is_snp = (hdf['variants']['VARIANT_TYPE'][:][is_pass] == b'SNP')
    is_bi = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'BI')
    is_sd = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'SD')
    is_mu = (hdf['variants']['MULTIALLELIC'][:][is_pass] == b'MU')
    is_ins = ((svlen1 > 0) | (svlen2 > 0))[is_pass]
    is_del = ((svlen1 < 0) | (svlen2 < 0))[is_pass]
    is_ins_del_het = (((svlen1 > 0) & (svlen2 < 0)) | ((svlen1 < 0) & (svlen2 > 0)))[is_pass] # These are hets where one allele is insertion and the other allele is deletion (these are probably rare)
    is_coding = (hdf['variants']['CDS'][:][is_pass])
    is_vqslod6 = (hdf['variants']['VQSLOD'][:][is_pass] >= 6.0)
    is_vhq_snp = (is_vqslod6 & is_snp & is_bi & is_coding)
    is_nonsynonymous = (hdf['variants']['SNPEFF_EFFECT'][:][is_pass] == b'NON_SYNONYMOUS_CODING')
    is_synonymous = (hdf['variants']['SNPEFF_EFFECT'][:][is_pass] == b'SYNONYMOUS_CODING')
    is_frameshift = (hdf['variants']['SNPEFF_EFFECT'][:][is_pass] == b'FRAME_SHIFT')
    is_inframe = np.in1d(hdf['variants']['SNPEFF_EFFECT'][:][is_pass], [b'CODON_INSERTION', b'CODON_DELETION', b'CODON_CHANGE_PLUS_CODON_DELETION', b'CODON_CHANGE_PLUS_CODON_INSERTION'])

    is_singleton = (
        ((ac1 == 1) & (genotypes[:, 0, 0] > 0)) |
        ((ac2 == 1) & (genotypes[:, 0, 1] > 0)) |
        ((ac1 == 2) & (genotypes[:, 0, 0] > 0) & (genotypes[:, 0, 1] > 0))
    )[is_pass]
    
    is_pass_nonref = (is_pass & ((genotypes[:, 0, 0] > 0) | (genotypes[:, 0, 1] > 0)))
    is_biallelic_snp_nonref = (is_snp & is_bi &((genotypes_pass[:, 0, 0] > 0) | (genotypes_pass[:, 0, 1] > 0)))
    is_biallelic_indel_nonref = (~is_snp & is_bi &((genotypes_pass[:, 0, 0] > 0) | (genotypes_pass[:, 0, 1] > 0)))
    
    GQ = hdf['calldata']['GQ'][:, [index]][is_pass]
    DP = hdf['calldata']['DP'][:, [index]][is_pass]
    PGT = hdf['calldata']['PGT'][:, [index]][is_pass]
    
    mutations = np.char.add(hdf['variants']['REF'][:][is_pass][is_biallelic_snp_nonref], hdf['variants']['ALT'][:, 0][is_pass][is_biallelic_snp_nonref])
    is_transition = np.in1d(mutations, [b'AG', b'GA', b'CT', b'TC'])
    is_transversion = np.in1d(mutations, [b'AC', b'AT', b'GC', b'GT', b'CA', b'CG', b'TA', b'TG'])
    is_AT_to_AT = np.in1d(mutations, [b'AT', b'TA'])
    is_CG_to_CG = np.in1d(mutations, [b'CG', b'GC'])
    is_AT_to_CG = np.in1d(mutations, [b'AC', b'AG', b'TC', b'TG'])
    is_CG_to_AT = np.in1d(mutations, [b'CA', b'GA', b'CT', b'GT'])

    results['num_variants']             = genotypes.shape[0]
    results['num_pass_variants']        = np.count_nonzero(is_pass)
    results['num_missing']              = genotypes.count_missing(axis=0)[0]
    results['num_pass_missing']         = genotypes_pass.count_missing(axis=0)[0]
    results['num_called']               = (results['num_variants'] - results['num_missing'])
    results['num_pass_called']          = (results['num_pass_variants'] - results['num_pass_missing'])

    results['num_het']                  = genotypes.count_het(axis=0)[0]
    results['num_pass_het']             = genotypes_pass.count_het(axis=0)[0]
    results['num_hom_alt']              = genotypes.count_hom_alt(axis=0)[0]
    results['num_pass_hom_alt']         = genotypes_pass.count_hom_alt(axis=0)[0]
#     results['num_pass_non_ref']         = (results['num_pass_het'] + results['num_pass_hom_alt'])
    results['num_pass_non_ref']         = np.count_nonzero(is_pass_nonref)
    
    results['num_biallelic_het']        = genotypes_pass.subset(is_bi).count_het(axis=0)[0]
    results['num_biallelic_hom_alt']    = genotypes_pass.subset(is_bi).count_hom_alt(axis=0)[0]
    results['num_spanning_del_het']     = genotypes_pass.subset(is_sd).count_het(axis=0)[0]
    results['num_spanning_del_hom_alt'] = genotypes_pass.subset(is_sd).count_hom_alt(axis=0)[0]
    results['num_multiallelic_het']     = genotypes_pass.subset(is_mu).count_het(axis=0)[0]
    results['num_multiallelic_hom_alt'] = genotypes_pass.subset(is_mu).count_hom_alt(axis=0)[0]
    
    results['num_snp_hom_ref']          = genotypes_pass.subset(is_snp).count_hom_ref(axis=0)[0]
    results['num_snp_het']              = genotypes_pass.subset(is_snp).count_het(axis=0)[0]
    results['num_snp_hom_alt']          = genotypes_pass.subset(is_snp).count_hom_alt(axis=0)[0]
    results['num_indel_hom_ref']        = genotypes_pass.subset(~is_snp).count_hom_ref(axis=0)[0]
    results['num_indel_het']            = genotypes_pass.subset(~is_snp).count_het(axis=0)[0]
    results['num_indel_hom_alt']        = genotypes_pass.subset(~is_snp).count_hom_alt(axis=0)[0]

    results['num_ins_het']              = genotypes_pass.subset(is_ins).count_het(axis=0)[0]
    results['num_ins_hom_alt']          = genotypes_pass.subset(is_ins).count_hom_alt(axis=0)[0]
    results['num_ins']                  = (results['num_ins_hom_alt'] + results['num_ins_het'])
    results['num_del_het']              = genotypes_pass.subset(is_del).count_het(axis=0)[0]
    results['num_del_hom_alt']          = genotypes_pass.subset(is_del).count_hom_alt(axis=0)[0]
    results['num_del']                  = (results['num_del_hom_alt'] + results['num_del_het'])
    
    results['num_coding_het']           = genotypes_pass.subset(is_coding).count_het(axis=0)[0]
    results['num_coding_hom_alt']       = genotypes_pass.subset(is_coding).count_hom_alt(axis=0)[0]
    results['num_coding']               = (results['num_coding_het'] + results['num_coding_hom_alt'])
    
    results['num_vhq_snp_hom_ref']      = genotypes_pass.subset(is_vhq_snp).count_hom_ref(axis=0)[0]
    results['num_vhq_snp_het']          = genotypes_pass.subset(is_vhq_snp).count_het(axis=0)[0]
    results['num_vhq_snp_hom_alt']      = genotypes_pass.subset(is_vhq_snp).count_hom_alt(axis=0)[0]
    
    results['num_singleton']            = np.count_nonzero(is_singleton)
    results['num_biallelic_singleton']  = np.count_nonzero(is_bi & is_singleton)
    results['num_vhq_snp_singleton']    = np.count_nonzero(is_vhq_snp & is_singleton)

    results['num_bi_nonsynonymous']     = np.count_nonzero(is_biallelic_snp_nonref & is_nonsynonymous)
    results['num_bi_synonymous']        = np.count_nonzero(is_biallelic_snp_nonref & is_synonymous)
    results['num_bi_frameshift']        = np.count_nonzero(is_biallelic_indel_nonref & is_frameshift)
    results['num_bi_inframe']           = np.count_nonzero(is_biallelic_indel_nonref & is_inframe)

    results['num_bi_transition']        = np.count_nonzero(is_transition)
    results['num_bi_transversion']      = np.count_nonzero(is_transversion)
    results['num_bi_AT_to_AT']          = np.count_nonzero(is_AT_to_AT)
    results['num_bi_CG_to_CG']          = np.count_nonzero(is_CG_to_CG)
    results['num_bi_AT_to_CG']          = np.count_nonzero(is_AT_to_CG)
    results['num_bi_CG_to_AT']          = np.count_nonzero(is_CG_to_AT)

    results['pc_pass']                  = results['num_pass_called'] / results['num_called']
    results['pc_missing']               = results['num_missing'] / results['num_variants']
    results['pc_pass_missing']          = results['num_pass_missing'] / results['num_pass_variants']
    results['pc_het']                   = results['num_het'] / results['num_called']
    results['pc_pass_het']              = results['num_pass_het'] / results['num_pass_called']
    results['pc_hom_alt']               = results['num_hom_alt'] / results['num_called']
    results['pc_pass_hom_alt']          = results['num_pass_hom_alt'] / results['num_pass_called']
    results['pc_snp']                   = (results['num_snp_het'] + results['num_snp_hom_alt']) / results['num_pass_non_ref']
#     results['pc_snp_v2']                = (results['num_snp_het'] + results['num_snp_hom_alt']) / (results['num_snp_het'] + results['num_snp_hom_alt'] + results['num_indel_het'] + results['num_indel_hom_alt'])
    results['pc_biallelic']             = (results['num_biallelic_het'] + results['num_biallelic_hom_alt']) / results['num_pass_non_ref']
    results['pc_spanning_del']          = (results['num_spanning_del_het'] + results['num_spanning_del_hom_alt']) / results['num_pass_non_ref']
    results['pc_mutliallelic']          = (results['num_multiallelic_het'] + results['num_multiallelic_hom_alt']) / results['num_pass_non_ref']
    results['pc_ins']                   = (results['num_ins'] / (results['num_ins'] + results['num_del']))
    results['pc_coding']                = results['num_coding'] / results['num_pass_non_ref']
#     results['pc_bi_nonsynonymous']      = results['num_bi_nonsynonymous'] / (results['num_biallelic_het'] + results['num_biallelic_hom_alt'])
    results['pc_bi_nonsynonymous']      = results['num_bi_nonsynonymous'] / (results['num_bi_nonsynonymous'] + results['num_bi_synonymous'])
    results['pc_bi_frameshift']         = results['num_bi_frameshift'] / (results['num_bi_frameshift'] + results['num_bi_inframe'])
    results['pc_bi_transition']         = results['num_bi_transition'] / (results['num_bi_transition'] + results['num_bi_transversion'])
    results['pc_bi_AT_to_AT']           = results['num_bi_AT_to_AT'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_CG_to_CG']           = results['num_bi_CG_to_CG'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_AT_to_CG']           = results['num_bi_AT_to_CG'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_CG_to_AT']           = results['num_bi_CG_to_AT'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    
    results['mean_GQ']                  = np.mean(GQ)
    results['mean_GQ_2']                = np.nanmean(GQ)
    results['mean_DP']                  = np.mean(DP)
    results['mean_DP_2']                = np.nanmean(DP)
    
    print('\t'.join([str(x) for x in list(results.keys())]))
    print('\t'.join([str(x) for x in list(results.values())]))

    return(results, PGT)

import allel
def create_sample_summary_2(index=0, hdf5_fn='/lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161128_HDF5_build/hdf5/Pf_60_npy_PID_a12.h5'):
    results = collections.OrderedDict()
    
    hdf = h5py.File(hdf5_fn, 'r')
    samples = hdf['samples'][:]
    results['sample_id'] = samples[index].decode('ascii')
#     output_fn = "%s/%s.txt" % (output_filestem, results['sample_id'])
#     fo = open(output_fn, 'w')
    is_pass = hdf['variants']['FILTER_PASS'][:]
    
    genotypes = allel.GenotypeArray(hdf['calldata']['genotype'][:, [index], :])
#     genotypes_pass = genotypes[is_pass]
    
    svlen = hdf['variants']['svlen'][:]
    svlen1 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 0] - 1]
    svlen1[np.in1d(genotypes[:, 0, 0], [-1, 0])] = 0 # Ref and missing are considered non-SV
    svlen2 = svlen[np.arange(svlen.shape[0]), genotypes[:, 0, 1] - 1]
    svlen2[np.in1d(genotypes[:, 0, 1], [-1, 0])] = 0 # Ref and missing are considered non-SV
    indel_len = svlen1
    het_indels = (svlen1 != svlen2)
    indel_len[het_indels] = svlen1[het_indels] + svlen2[het_indels]
    
    is_indel = (indel_len != 0)
    is_inframe = ((indel_len != 0) & (indel_len%3 == 0))
    is_frameshift = ((indel_len != 0) & (indel_len%3 != 0))
    
    ac = hdf['variants']['AC'][:]
    ac1 = ac[np.arange(ac.shape[0]), genotypes[:, 0, 0] - 1]
    ac1[np.in1d(genotypes[:, 0, 0], [-1, 0])] = 0 # Ref and missing are considered non-singleton
    ac2 = ac[np.arange(ac.shape[0]), genotypes[:, 0, 1] - 1]
    ac2[np.in1d(genotypes[:, 0, 1], [-1, 0])] = 0 # Ref and missing are considered non-singleton
    
    is_snp = (hdf['variants']['VARIANT_TYPE'][:] == b'SNP')
    is_bi = (hdf['variants']['MULTIALLELIC'][:] == b'BI')
    is_sd = (hdf['variants']['MULTIALLELIC'][:] == b'SD')
    is_mu = (hdf['variants']['MULTIALLELIC'][:] == b'MU')
    is_ins = ((svlen1 > 0) | (svlen2 > 0))
    is_del = ((svlen1 < 0) | (svlen2 < 0))
    is_ins_del_het = (((svlen1 > 0) & (svlen2 < 0)) | ((svlen1 < 0) & (svlen2 > 0))) # These are hets where one allele is insertion and the other allele is deletion (these are probably rare)
    is_coding = (hdf['variants']['CDS'][:])
    is_vqslod6 = (hdf['variants']['VQSLOD'][:] >= 6.0)
    is_hq_snp = (is_pass & is_snp & is_bi & is_coding)
    is_vhq_snp = (is_pass & is_vqslod6 & is_snp & is_bi & is_coding)
    is_nonsynonymous = (hdf['variants']['SNPEFF_EFFECT'][:] == b'NON_SYNONYMOUS_CODING')
    is_synonymous = (hdf['variants']['SNPEFF_EFFECT'][:] == b'SYNONYMOUS_CODING')
    is_frameshift_snpeff = (hdf['variants']['SNPEFF_EFFECT'][:] == b'FRAME_SHIFT')
    is_inframe_snpeff = np.in1d(hdf['variants']['SNPEFF_EFFECT'][:], [b'CODON_INSERTION', b'CODON_DELETION', b'CODON_CHANGE_PLUS_CODON_DELETION', b'CODON_CHANGE_PLUS_CODON_INSERTION'])

    is_singleton = (
        ((ac1 == 1) & (genotypes[:, 0, 0] > 0)) |
        ((ac2 == 1) & (genotypes[:, 0, 1] > 0)) |
        ((ac1 == 2) & (genotypes[:, 0, 0] > 0) & (genotypes[:, 0, 1] > 0))
    )
    
    is_hom_ref = ((genotypes[:, 0, 0] == 0) & (genotypes[:, 0, 1] == 0))
    is_het = ((genotypes[:, 0, 0] != genotypes[:, 0, 1]))
    is_hom_alt = ((genotypes[:, 0, 0] > 0) & (genotypes[:, 0, 0] == genotypes[:, 0, 1]))
    is_non_ref = ((genotypes[:, 0, 0] > 0) | (genotypes[:, 0, 1] > 0))
    is_missing = ((genotypes[:, 0, 0] == -1))
    is_called = ((genotypes[:, 0, 0] >= 0))
    
    GQ = hdf['calldata']['GQ'][:, index]
    is_GQ_30 = (GQ >= 30)
    is_GQ_99 = (GQ >= 99)
    DP = hdf['calldata']['DP'][:, index]
    PGT = hdf['calldata']['PGT'][:, index]
    is_phased = np.in1d(PGT, [b'.', b''], invert=True)
    
    mutations = np.char.add(hdf['variants']['REF'][:][(is_pass & is_snp & is_bi & is_non_ref)], hdf['variants']['ALT'][:, 0][(is_pass & is_snp & is_bi & is_non_ref)])
    is_transition = np.in1d(mutations, [b'AG', b'GA', b'CT', b'TC'])
    is_transversion = np.in1d(mutations, [b'AC', b'AT', b'GC', b'GT', b'CA', b'CG', b'TA', b'TG'])
    is_AT_to_AT = np.in1d(mutations, [b'AT', b'TA'])
    is_CG_to_CG = np.in1d(mutations, [b'CG', b'GC'])
    is_AT_to_CG = np.in1d(mutations, [b'AC', b'AG', b'TC', b'TG'])
    is_CG_to_AT = np.in1d(mutations, [b'CA', b'GA', b'CT', b'GT'])

    results['num_variants']             = genotypes.shape[0]
    results['num_pass_variants']        = np.count_nonzero(is_pass)
    results['num_missing']              = np.count_nonzero(is_missing)
    results['num_pass_missing']         = np.count_nonzero(is_pass & is_missing)
    results['num_called']               = np.count_nonzero(~is_missing)
#     results['num_called']               = (results['num_variants'] - results['num_missing'])
    results['num_pass_called']          = np.count_nonzero(is_pass & is_called)
#     results['num_pass_called_2']        = np.count_nonzero(is_pass & ~is_missing)
#     results['num_pass_called']          = (results['num_pass_variants'] - results['num_pass_missing'])

    results['num_hom_ref']              = np.count_nonzero(is_hom_ref)
    results['num_het']                  = np.count_nonzero(is_het)
    results['num_pass_het']             = np.count_nonzero(is_pass & is_het)
    results['num_hom_alt']              = np.count_nonzero(is_hom_alt)
    results['num_pass_hom_alt']         = np.count_nonzero(is_pass & is_hom_alt)
#     results['num_pass_non_ref']         = (results['num_pass_het'] + results['num_pass_hom_alt'])
    results['num_pass_non_ref']         = np.count_nonzero(is_pass & is_non_ref)
#     results['num_variants_2']           = results['num_hom_ref'] + results['num_het'] + results['num_hom_alt'] + results['num_missing']
    
    results['num_biallelic_het']        = np.count_nonzero(is_pass & is_bi & is_het)
    results['num_biallelic_hom_alt']    = np.count_nonzero(is_pass & is_bi & is_hom_alt)
    results['num_spanning_del_het']     = np.count_nonzero(is_pass & is_sd & is_het)
    results['num_spanning_del_hom_alt'] = np.count_nonzero(is_pass & is_sd & is_hom_alt)
    results['num_multiallelic_het']     = np.count_nonzero(is_pass & is_mu & is_het)
    results['num_multiallelic_hom_alt'] = np.count_nonzero(is_pass & is_mu & is_hom_alt)
    
#     results['num_snp_hom_ref']          = np.count_nonzero(is_pass & is_snp & is_het)
    results['num_snp_het']              = np.count_nonzero(is_pass & is_snp & is_het)
    results['num_snp_hom_alt']          = np.count_nonzero(is_pass & is_snp & is_hom_alt)
    results['num_snp']                  = (results['num_snp_het'] + results['num_snp_hom_alt'])
#     results['num_indel_hom_ref']        = genotypes_pass.subset(~is_snp).count_hom_ref(axis=0)[0]
    results['num_indel_het']            = np.count_nonzero(is_pass & ~is_snp & is_het)
    results['num_indel_hom_alt']        = np.count_nonzero(is_pass & ~is_snp & is_het)
    results['num_indel']                  = (results['num_indel_het'] + results['num_indel_hom_alt'])

    results['num_ins_het']              = np.count_nonzero(is_pass & is_ins & is_het)
    results['num_ins_hom_alt']          = np.count_nonzero(is_pass & is_ins & is_hom_alt)
    results['num_ins']                  = (results['num_ins_hom_alt'] + results['num_ins_het'])
    results['num_del_het']              = np.count_nonzero(is_pass & is_del & is_het)
    results['num_del_hom_alt']          = np.count_nonzero(is_pass & is_del & is_hom_alt)
    results['num_del']                  = (results['num_del_hom_alt'] + results['num_del_het'])
    
    results['num_coding_het']           = np.count_nonzero(is_pass & is_coding & is_het)
    results['num_coding_hom_alt']       = np.count_nonzero(is_pass & is_coding & is_hom_alt)
    results['num_coding']               = (results['num_coding_het'] + results['num_coding_hom_alt'])
    
    results['num_hq_snp_called']        = np.count_nonzero(is_hq_snp & ~is_missing)
    results['num_hq_snp_hom_ref']       = np.count_nonzero(is_hq_snp & is_hom_ref)
    results['num_hq_snp_het']           = np.count_nonzero(is_hq_snp & is_het)
    results['num_hq_snp_hom_alt']       = np.count_nonzero(is_hq_snp & is_hom_alt)
    results['num_vhq_snp_called']       = np.count_nonzero(is_vhq_snp & ~is_missing)
    results['num_vhq_snp_hom_ref']      = np.count_nonzero(is_vhq_snp & is_hom_ref)
    results['num_vhq_snp_het']          = np.count_nonzero(is_vhq_snp & is_het)
    results['num_vhq_snp_hom_alt']      = np.count_nonzero(is_vhq_snp & is_hom_alt)
    
    results['num_singleton']            = np.count_nonzero(is_pass & is_singleton)
    results['num_biallelic_singleton']  = np.count_nonzero(is_pass & is_bi & is_singleton)
    results['num_hq_snp_singleton']     = np.count_nonzero(is_hq_snp & is_singleton)
    results['num_vhq_snp_singleton']    = np.count_nonzero(is_vhq_snp & is_singleton)

    results['num_bi_nonsynonymous']     = np.count_nonzero(is_pass & is_bi & is_snp & is_non_ref & is_nonsynonymous)
    results['num_bi_synonymous']        = np.count_nonzero(is_pass & is_bi & is_snp & is_non_ref & is_synonymous)
#     results['num_frameshift']           = np.count_nonzero(is_pass & is_indel & is_non_ref & is_coding & is_frameshift)
#     results['num_inframe']              = np.count_nonzero(is_pass & is_indel & is_non_ref & is_coding & is_inframe)
    results['num_frameshift']           = np.count_nonzero(is_pass & is_indel & is_coding & is_frameshift)
    results['num_inframe']              = np.count_nonzero(is_pass & is_indel & is_coding & is_inframe)
    results['num_bi_frameshift']        = np.count_nonzero(is_pass & is_bi & is_indel & is_coding & is_non_ref & is_frameshift)
    results['num_bi_inframe']           = np.count_nonzero(is_pass & is_bi & is_indel & is_coding & is_non_ref & is_inframe)
    results['num_hq_frameshift']        = np.count_nonzero(is_pass & is_vqslod6 & is_bi & is_indel & is_coding & is_non_ref & is_frameshift)
    results['num_hq_inframe']           = np.count_nonzero(is_pass & is_vqslod6 & is_bi & is_indel & is_coding & is_non_ref & is_inframe)
    results['num_bi_frameshift_snpeff'] = np.count_nonzero(is_pass & is_bi & ~is_snp & is_non_ref & is_frameshift_snpeff)
    results['num_bi_inframe_snpeff']    = np.count_nonzero(is_pass & is_bi & ~is_snp & is_non_ref & is_inframe_snpeff)

    results['num_bi_transition']        = np.count_nonzero(is_transition)
    results['num_bi_transversion']      = np.count_nonzero(is_transversion)
    results['num_bi_AT_to_AT']          = np.count_nonzero(is_AT_to_AT)
    results['num_bi_CG_to_CG']          = np.count_nonzero(is_CG_to_CG)
    results['num_bi_AT_to_CG']          = np.count_nonzero(is_AT_to_CG)
    results['num_bi_CG_to_AT']          = np.count_nonzero(is_CG_to_AT)

    results['num_phased']               = np.count_nonzero(is_pass & is_phased)
    results['num_phased_non_ref']       = np.count_nonzero(is_pass & is_phased & is_non_ref)
    results['num_phased_hom_ref']       = np.count_nonzero(is_pass & is_phased & is_hom_ref)
    results['num_phased_missing']       = np.count_nonzero(is_pass & is_phased & is_missing)
    
    results['num_GQ_30']                = np.count_nonzero(is_pass & is_called & is_GQ_30)
    results['num_het_GQ_30']            = np.count_nonzero(is_pass & is_het & is_GQ_30)
    results['num_hom_alt_GQ_30']        = np.count_nonzero(is_pass & is_hom_alt & is_GQ_30)
    results['num_GQ_99']                = np.count_nonzero(is_pass & is_called & is_GQ_99)
    results['num_het_GQ_99']            = np.count_nonzero(is_pass & is_het & is_GQ_99)
    results['num_hom_alt_GQ_99']        = np.count_nonzero(is_pass & is_hom_alt & is_GQ_99)

    results['pc_pass']                  = 0.0 if results['num_called'] == 0 else         results['num_pass_called'] / results['num_called']
    results['pc_missing']               = 0.0 if results['num_variants'] == 0 else         results['num_missing'] / results['num_variants']
    results['pc_pass_missing']          = 0.0 if results['num_pass_variants'] == 0 else         results['num_pass_missing'] / results['num_pass_variants']
    results['pc_het']                   = 0.0 if results['num_called'] == 0 else         results['num_het'] / results['num_called']
    results['pc_pass_het']              = 0.0 if results['num_pass_called'] == 0 else         results['num_pass_het'] / results['num_pass_called']
    results['pc_hq_snp_het']            = 0.0 if results['num_hq_snp_called'] == 0 else         results['num_hq_snp_het'] / results['num_hq_snp_called']
    results['pc_vhq_snp_het']           = 0.0 if results['num_vhq_snp_called'] == 0 else         results['num_vhq_snp_het'] / results['num_vhq_snp_called']
    results['pc_hom_alt']               = 0.0 if results['num_called'] == 0 else         results['num_hom_alt'] / results['num_called']
    results['pc_pass_hom_alt']          = 0.0 if results['num_pass_called'] == 0 else         results['num_pass_hom_alt'] / results['num_pass_called']
    results['pc_snp']                   = 0.0 if results['num_pass_non_ref'] == 0 else         (results['num_snp_het'] + results['num_snp_hom_alt']) / results['num_pass_non_ref']
#     results['pc_snp_v2']                = (results['num_snp_het'] + results['num_snp_hom_alt']) / (results['num_snp_het'] + results['num_snp_hom_alt'] + results['num_indel_het'] + results['num_indel_hom_alt'])
    results['pc_biallelic']             = 0.0 if results['num_pass_non_ref'] == 0 else         (results['num_biallelic_het'] + results['num_biallelic_hom_alt']) / results['num_pass_non_ref']
    results['pc_spanning_del']          = 0.0 if results['num_pass_non_ref'] == 0 else         (results['num_spanning_del_het'] + results['num_spanning_del_hom_alt']) / results['num_pass_non_ref']
    results['pc_mutliallelic']          = 0.0 if results['num_pass_non_ref'] == 0 else         (results['num_multiallelic_het'] + results['num_multiallelic_hom_alt']) / results['num_pass_non_ref']
    results['pc_ins']                   = 0.0 if (results['num_ins'] + results['num_del']) == 0 else         (results['num_ins'] / (results['num_ins'] + results['num_del']))
    results['pc_coding']                = 0.0 if results['num_pass_non_ref'] == 0 else         results['num_coding'] / results['num_pass_non_ref']
#     results['pc_bi_nonsynonymous']      = results['num_bi_nonsynonymous'] / (results['num_biallelic_het'] + results['num_biallelic_hom_alt'])
    results['pc_bi_nonsynonymous']      = 0.0 if (results['num_bi_nonsynonymous'] + results['num_bi_synonymous']) == 0 else         results['num_bi_nonsynonymous'] / (results['num_bi_nonsynonymous'] + results['num_bi_synonymous'])
    results['pc_frameshift']            = 0.0 if (results['num_frameshift'] + results['num_inframe']) == 0 else         results['num_frameshift'] / (results['num_frameshift'] + results['num_inframe'])
    results['pc_bi_frameshift']         = 0.0 if (results['num_bi_frameshift'] + results['num_bi_inframe']) == 0 else         results['num_bi_frameshift'] / (results['num_bi_frameshift'] + results['num_bi_inframe'])
    results['pc_hq_frameshift']         = 0.0 if (results['num_hq_frameshift'] + results['num_hq_inframe']) == 0 else         results['num_hq_frameshift'] / (results['num_hq_frameshift'] + results['num_hq_inframe'])
    results['pc_bi_frameshift_snpeff']  = 0.0 if (results['num_bi_frameshift_snpeff'] + results['num_bi_inframe_snpeff']) == 0 else         results['num_bi_frameshift_snpeff'] / (results['num_bi_frameshift_snpeff'] + results['num_bi_inframe_snpeff'])
    results['pc_bi_transition']         = 0.0 if (results['num_bi_transition'] + results['num_bi_transversion']) == 0 else         results['num_bi_transition'] / (results['num_bi_transition'] + results['num_bi_transversion'])
    results['pc_bi_AT_to_AT']           = 0.0 if (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT']) == 0 else         results['num_bi_AT_to_AT'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_CG_to_CG']           = 0.0 if (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT']) == 0 else         results['num_bi_CG_to_CG'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_AT_to_CG']           = 0.0 if (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT']) == 0 else         results['num_bi_AT_to_CG'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_bi_CG_to_AT']           = 0.0 if (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT']) == 0 else         results['num_bi_CG_to_AT'] / (results['num_bi_AT_to_AT'] + results['num_bi_CG_to_CG'] + results['num_bi_AT_to_CG'] + results['num_bi_CG_to_AT'])
    results['pc_phased']                = 0.0 if results['num_pass_non_ref'] == 0 else         results['num_phased_non_ref'] / results['num_pass_non_ref']
    results['pc_phased_hom_ref']        = 0.0 if results['num_phased'] == 0 else         results['num_phased_hom_ref'] / results['num_phased']
    results['pc_phased_missing']        = 0.0 if results['num_phased'] == 0 else         results['num_phased_missing'] / results['num_phased']
    results['pc_GQ_30']                 = 0.0 if results['num_pass_called'] == 0 else         results['num_GQ_30'] / results['num_pass_called']
    results['pc_het_GQ_30']             = 0.0 if results['num_pass_het'] == 0 else         results['num_het_GQ_30'] / results['num_pass_het']
    results['pc_hom_alt_GQ_30']         = 0.0 if results['num_pass_hom_alt'] == 0 else         results['num_hom_alt_GQ_30'] / results['num_pass_hom_alt']
    results['pc_GQ_99']                 = 0.0 if results['num_pass_called'] == 0 else         results['num_GQ_99'] / results['num_pass_called']
    results['pc_het_GQ_99']             = 0.0 if results['num_pass_het'] == 0 else         results['num_het_GQ_99'] / results['num_pass_het']
    results['pc_hom_alt_GQ_99']         = 0.0 if results['num_pass_hom_alt'] == 0 else         results['num_hom_alt_GQ_99'] / results['num_pass_hom_alt']
     
    results['mean_GQ']                  = np.mean(GQ[is_pass])
    results['mean_GQ_hom_ref']          = np.mean(GQ[is_pass & is_hom_ref])
    results['mean_GQ_het']              = np.mean(GQ[is_pass & is_het])
    results['mean_GQ_hom_alt']          = np.mean(GQ[is_pass & is_hom_alt])
    results['mean_DP']                  = np.mean(DP[is_pass])
    results['mean_DP_hom_ref']          = np.mean(DP[is_pass & is_hom_ref])
    results['mean_DP_het']              = np.mean(DP[is_pass & is_het])
    results['mean_DP_hom_alt']          = np.mean(DP[is_pass & is_hom_alt])
#     results['mean_GQ_2']                = np.nanmean(GQ[is_pass])
#     results['mean_DP_2']                = np.nanmean(DP[is_pass])
#     results['mean_DP']                  = np.mean(DP)
#     results['mean_DP_2']                = np.nanmean(DP)

    results['mean_indel_len']           = np.mean(indel_len[is_pass])
    results['total_indel_len']          = np.sum(indel_len[is_pass])

    print('\t'.join([str(x) for x in list(results.keys())]))
    print('\t'.join([str(x) for x in list(results.values())]))

#     return(results, is_pass, is_phased, is_non_ref, is_hom_ref, is_missing)
#     return(results, svlen, svlen1, svlen2, indel_len, is_indel, is_inframe, is_frameshift, is_pass, is_bi, is_non_ref, is_frameshift_snpeff, is_inframe_snpeff, is_coding, is_vqslod6)
#     return(results, is_pass, is_called, is_GQ_30)
    return(results)

results = create_sample_summary_2(index=275)

results, is_pass, is_called, is_GQ_30 = create_sample_summary_2()

print('pc_GQ_30', results['pc_GQ_30'])
print('pc_het_GQ_30', results['pc_het_GQ_30'])
print('pc_hom_alt_GQ_30', results['pc_hom_alt_GQ_30'])
print('pc_GQ_99', results['pc_GQ_99'])
print('pc_het_GQ_99', results['pc_het_GQ_99'])
print('pc_hom_alt_GQ_99', results['pc_hom_alt_GQ_99'])

results = create_sample_summary_2()

print('num_frameshift', results['num_frameshift'])
print('num_inframe', results['num_inframe'])
print('num_bi_frameshift', results['num_bi_frameshift'])
print('num_bi_inframe', results['num_bi_inframe'])
print('num_hq_frameshift', results['num_bi_frameshift'])
print('num_hq_inframe', results['num_bi_inframe'])
print('num_bi_frameshift_snpeff', results['num_bi_frameshift_snpeff'])
print('num_bi_inframe_snpeff', results['num_bi_inframe_snpeff'])
print()
print('pc_frameshift', results['pc_frameshift'])
print('pc_bi_frameshift', results['pc_bi_frameshift'])
print('pc_hq_frameshift', results['pc_bi_frameshift'])
print('pc_bi_frameshift_snpeff', results['pc_bi_frameshift_snpeff'])

np.count_nonzero(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift)

np.where(is_pass & is_vqslod6 & is_bi & is_indel & is_non_ref & is_coding & is_frameshift)[0]

np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift)[0]

np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift_snpeff)[0]

np.setdiff1d(
    np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift)[0],
    np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift_snpeff)[0]
)

np.setdiff1d(
    np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift_snpeff)[0],
    np.where(is_pass & is_bi & is_indel & is_non_ref & is_coding & is_frameshift)[0]
)

index=140202
print(hdf['Pf60']['variants']['REF'][index])
print(hdf['Pf60']['variants']['ALT'][index, :])
print(hdf['Pf60']['variants']['CDS'][index])
print(hdf['Pf60']['variants']['SNPEFF_EFFECT'][index])
print(hdf['Pf60']['calldata']['genotype'][index, 0, :])
print(hdf['Pf60']['calldata']['AD'][index, 0, :])
print(hdf['Pf60']['calldata']['PL'][index, 0, :])
print(hdf['Pf60']['calldata']['PID'][index, 0])
print(hdf['Pf60']['calldata']['PGT'][index, 0])

index=61186
print(hdf['Pf60']['variants']['REF'][index])
print(hdf['Pf60']['variants']['ALT'][index, :])
print(hdf['Pf60']['variants']['SNPEFF_EFFECT'][index])
print(hdf['Pf60']['calldata']['genotype'][index, 0, :])
print(hdf['Pf60']['calldata']['AD'][index, 0, :])
print(hdf['Pf60']['calldata']['PL'][index, 0, :])
print(hdf['Pf60']['calldata']['PID'][index, 0])
print(hdf['Pf60']['calldata']['PGT'][index, 0])

is_phased_hom_ref = (is_pass & is_phased & is_hom_ref)
is_phased_missing = (is_pass & is_phased & is_missing)
print(np.count_nonzero(is_phased_hom_ref))
print(np.count_nonzero(is_phased_missing))

np.where(is_phased_hom_ref)

index=71926
print(hdf['Pf60']['variants']['REF'][index])
print(hdf['Pf60']['variants']['ALT'][index, :])
print(hdf['Pf60']['calldata']['genotype'][index, 0, :])
print(hdf['Pf60']['calldata']['AD'][index, 0, :])
print(hdf['Pf60']['calldata']['PL'][index, 0, :])
print(hdf['Pf60']['calldata']['PID'][index, 0])
print(hdf['Pf60']['calldata']['PGT'][index, 0])

index=79318
print(hdf['Pf60']['variants']['REF'][index])
print(hdf['Pf60']['variants']['ALT'][index, :])
print(hdf['Pf60']['calldata']['genotype'][index, 0, :])
print(hdf['Pf60']['calldata']['AD'][index, 0, :])
print(hdf['Pf60']['calldata']['PL'][index, 0, :])
print(hdf['Pf60']['calldata']['PID'][index, 0])
print(hdf['Pf60']['calldata']['PGT'][index, 0])

index=79318
print(hdf['Pf60']['variants']['REF'][71926])
print(hdf['Pf60']['variants']['ALT'][71926, :])
print(hdf['Pf60']['calldata']['genotype'][71926, 0, :])
print(hdf['Pf60']['calldata']['AD'][71926, 0, :])
print(hdf['Pf60']['calldata']['PL'][71926, 0, :])
print(hdf['Pf60']['calldata']['PID'][71926, 0])
print(hdf['Pf60']['calldata']['PGT'][71926, 0])

hdf['Pf60']['variants']['ALT'][:][71926, :]

hdf['Pf60']['calldata']['genotype'][:, 0, :][71926, :]

hdf['Pf60']['calldata']['genotype'][71926, 0, :]

hdf['Pf60']['calldata']['AD'][71926, 0, :]

hdf['Pf60']['calldata']['PL'][71926, 0, :]

hdf['Pf60']['calldata']['PID'][71926, 0]

hdf['Pf60']['calldata']['PGT'][71926, 0]

np.where(is_phased_missing)

pd.value_counts(PGT[:,0])

results, mutations = create_sample_summary()

is_pass = hdf['Pf60']['variants']['FILTER_PASS'][:]
is_snp = (hdf['Pf60']['variants']['VARIANT_TYPE'][:][is_pass] == b'SNP')
is_bi = (hdf['Pf60']['variants']['MULTIALLELIC'][:][is_pass] == b'BI')
temp = hdf['Pf60']['variants']['REF'][:][is_pass][is_snp]

mutations = np.char.add(
    hdf['Pf60']['variants']['REF'][:][is_pass][(is_snp & is_bi)],
    hdf['Pf60']['variants']['ALT'][:, 0][is_pass][(is_snp & is_bi)]
)

pd.value_counts(mutations)

pd.value_counts(mutations)

list(hdf['Pf60'].keys())

pd.value_counts(hdf['Pf60']['variants']['AC'][:,0] == 0)

pd.value_counts(hdf['Pf60']['variants']['SNPEFF_EFFECT'][:])

7182/20

pd.value_counts(hdf['Pf60']['variants']['SNPEFF_EFFECT'][:][(
            hdf['Pf60']['variants']['FILTER_PASS'][:] &
            (hdf['Pf60']['variants']['MULTIALLELIC'][:] == b'BI') &
            (hdf['Pf60']['variants']['AC'][:, 0] > 359) &
            (hdf['Pf60']['variants']['AC'][:, 0] < (7182-359))
        )])

print('num_pass_non_ref', results['num_pass_non_ref'])
print('num_pass_non_ref_2', results['num_pass_non_ref_2'])

print('pc_bi_transition', results['pc_bi_transition'])
print('pc_bi_frameshift', results['pc_bi_frameshift'])
print('num_bi_frameshift', results['num_bi_frameshift'])
print('num_bi_inframe', results['num_bi_inframe'])
print('pc_biallelic', results['pc_biallelic'])
print('pc_spanning_del', results['pc_spanning_del'])
print('pc_mutliallelic', results['pc_mutliallelic'])
print('pc_bi_nonsynonymous', results['pc_bi_nonsynonymous'])
print('pc_bi_nonsynonymous_2', results['pc_bi_nonsynonymous_2'])

pd.value_counts(mutations)

print('num_snp_hom_ref', results['num_snp_hom_ref'])
print('num_snp_het', results['num_snp_het'])
print('num_snp_hom_alt', results['num_snp_hom_alt'])
print('num_indel_hom_ref', results['num_indel_hom_ref'])
print('num_indel_het', results['num_indel_het'])
print('num_indel_hom_alt', results['num_indel_hom_alt'])
print()
print('pc_snp', results['pc_snp'])
# print('pc_snp_v2', results['pc_snp_v2'])

create_sample_summary()

get_ipython().system('/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\\ 6.0/scripts/create_sample_summary.py')

fo = open(run_create_sample_summary_job_fn, 'w')
print('''HDF5_FN=$1
LSB_JOBINDEX=1
INDEX=$((LSB_JOBINDEX-1))
echo $INDEX

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\ 6.0/scripts/create_sample_summary.py \
--hdf5_fn $HDF5_FN --index $INDEX

''', file=fo)
fo.close()

fo = open(run_create_sample_summary_job_fn, 'w')
print('''HDF5_FN=$1
RELEASE=$2
# LSB_JOBINDEX=1
INDEX=$((LSB_JOBINDEX-1))
echo $INDEX
OUTPUT_FN=%s/sample_summaries/$RELEASE/results_$INDEX.txt

/nfs/team112_internal/rp7/src/github/malariagen/methods-dev/builds/Pf\ 6.0/scripts/create_sample_summary.py \
--hdf5_fn $HDF5_FN --index $INDEX > $OUTPUT_FN

''' % (output_dir), file=fo)
fo.close()

get_ipython().system('LSB_JOBINDEX=2 && bash {run_create_sample_summary_job_fn} /lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161128_HDF5_build/hdf5/Pf_60_npy_PID_a12.h5 Pf60')

MEMORY=8000
# Kick off Pf 6.0 jobs
get_ipython().system('bsub -q normal -G malaria-dk -J "summ[1-7182]" -n4 -R"select[mem>{MEMORY}] rusage[mem={MEMORY}] span[hosts=1]" -M {MEMORY} -o {"%s/log/output_%%J-%%I.log" % output_dir} bash {run_create_sample_summary_job_fn} /lustre/scratch111/malaria/rp7/data/methods-dev/builds/Pf6.0/20161128_HDF5_build/hdf5/Pf_60_npy_PID_a12.h5 Pf60')

MEMORY=8000
# Kick off Pv 3.0 jobs
get_ipython().system('bsub -q normal -G malaria-dk -J "summ[1-1001]" -n4 -R"select[mem>{MEMORY}] rusage[mem={MEMORY}] span[hosts=1]" -M {MEMORY} -o {"%s/log/output_%%J-%%I.log" % output_dir} bash {run_create_sample_summary_job_fn} {hdf_fn[\'Pv30\']} Pv30')

get_ipython().system("(head -n 1 {output_dir}/sample_summaries/Pf60/results_0.txt &   cat {output_dir}/sample_summaries/Pf60/results_*.txt | grep -v '^sample_id') > {output_dir}/pf_60_summaries.txt")

get_ipython().system("(head -n 1 {output_dir}/sample_summaries/Pv30/results_0.txt &   cat {output_dir}/sample_summaries/Pv30/results_*.txt | grep -v '^sample_id') > {output_dir}/pv_30_summaries.txt")

output_dir



create_sample_summary(1)

df_sample_summary, results = create_sample_summary()

df_sample_summary, results = create_sample_summary(index=100)

b'FP0008-C'.decode('ascii')

'\t'.join([str(x) for x in list(results.values())])

'\t'.join([str(x) for x in list(results.values())])

'\t'.join(list(results.values()))

list(results.values())

list(results.values())[0]

hdf = h5py.File(hdf_fn['Pv30'], 'r')
hdf['calldata']['genotype'].shape

svlen = hdf['variants']['svlen'][:]
svlen

genotype = hdf['calldata']['genotype'][:, 0, :]
genotype

pd.value_counts(genotype[:,0])

print(genotype.shape)
print(svlen.shape)

svlen1 = svlen[np.arange(svlen.shape[0]), genotype[:, 0] - 1]
svlen1[np.in1d(genotype[:, 0], [-1, 0])] = 0

pd.Series(svlen1).describe()

pd.value_counts(svlen1)

svlen2 = svlen[genotype]

svlen

genotype[:,0]

svlen[:, genotype[:,0]]

np.take(svlen[0:100000], genotype[0:100000,0]-1, axis=1)

alt_indexes = genotype[:, 0] - 1
alt_indexes[alt_indexes < 0] = 0

pd.value_counts(alt_indexes)

np.take(svlen[0:10000], alt_indexes[0:10000], axis=0).shape

svlen[0:1002]

alt_indexes[0:1002]

np.take(svlen[0:1002], alt_indexes[0:1002], axis=0)

svlen[0:1002][np.arange(1002), alt_indexes[0:1002]].shape

alt_indexes[0:10000].shape

svlen[0:10000].shape

svlen

print(svlen2.shape)
svlen2

temp = hdf['calldata']['genotype'][:, [0], :]

temp2=allel.GenotypeArray(temp)
temp2

temp.shape

get_ipython().run_cell_magic('time', '', "df_sample_summary = collections.OrderedDict()\n# for release in genotypes_subset:\nfor release in ['Pv30', 'Pf60']:\n    print(release)\n    samples = hdf[release]['samples'][:]\n    pass_variants = hdf[release]['variants']['FILTER_PASS'][:]\n    \n    print(0)\n    is_snp = (hdf[release]['variants']['VARIANT_TYPE'][:][pass_variants] == b'SNP')\n    is_bi = (hdf[release]['variants']['MULTIALLELIC'][:][pass_variants] == b'BI')\n    is_sd = (hdf[release]['variants']['MULTIALLELIC'][:][pass_variants] == b'SD')\n    is_mu = (hdf[release]['variants']['MULTIALLELIC'][:][pass_variants] == b'MU')\n    is_ins = (hdf[release]['variants']['svlen'][:][pass_variants] > 0)\n    is_del = (hdf[release]['variants']['svlen'][:][pass_variants] < 0)\n    \n    print(1)\n    num_variants = genotypes_subset[release]['all'].shape[0]\n    num_pass_variants = genotypes_subset[release]['pass'].shape[0]\n    num_missing = genotypes_subset[release]['all'].count_missing(axis=0)[:]\n    num_pass_missing = genotypes_subset[release]['pass'].count_missing(axis=0)[:]\n    num_called = (num_variants - num_missing)\n    num_pass_called = (num_pass_variants - num_pass_missing)\n    print(2)\n    num_het = genotypes_subset[release]['all'].count_het(axis=0)[:]\n    num_pass_het = genotypes_subset[release]['pass'].count_het(axis=0)[:]\n    num_hom_alt = genotypes_subset[release]['all'].count_hom_alt(axis=0)[:]\n    num_pass_hom_alt = genotypes_subset[release]['pass'].count_hom_alt(axis=0)[:]\n    print(3)\n    num_snp_hom_ref = genotypes_subset[release]['pass'].subset(is_snp).count_hom_ref(axis=0)[:]\n    num_snp_het = genotypes_subset[release]['pass'].subset(is_snp).count_het(axis=0)[:]\n    num_snp_hom_alt = genotypes_subset[release]['pass'].subset(is_snp).count_hom_alt(axis=0)[:]\n    num_indel_hom_ref = genotypes_subset[release]['pass'].subset(is_snp).count_hom_ref(axis=0)[:]\n    num_indel_het = genotypes_subset[release]['pass'].subset(is_snp).count_het(axis=0)[:]\n    num_indel_hom_alt = genotypes_subset[release]['pass'].subset(is_snp).count_hom_alt(axis=0)[:]\n    print(4)    \n    num_ins_hom_ref = genotypes_subset[release]['pass'].subset(is_ins).count_hom_ref(axis=0)[:]\n    num_ins_het = genotypes_subset[release]['pass'].subset(is_ins).count_het(axis=0)[:]\n    num_ins = (num_ins_hom_ref + num_ins_het)\n    num_del_hom_ref = genotypes_subset[release]['pass'].subset(is_del).count_hom_ref(axis=0)[:]\n    num_del_het = genotypes_subset[release]['pass'].subset(is_del).count_het(axis=0)[:]\n    num_del = (num_del_hom_ref + num_del_het)\n    \n    print(5)\n    pc_pass = num_pass_called / num_called\n    pc_missing = num_missing / num_variants\n    pc_pass_missing = num_pass_missing / num_pass_variants\n    pc_het = num_het / num_called\n    pc_pass_het = num_pass_het / num_pass_called\n    pc_hom_alt = num_hom_alt / num_called\n    pc_pass_hom_alt = num_pass_hom_alt / num_pass_called\n    pc_snp = (num_snp_het + num_snp_homalt) / (num_snp_het + num_snp_homalt + num_indel_het + num_indel_homalt)\n    pc_ins = (num_ins / (num_ins + num_del))\n\n    print(6)\n    df_sample_summary[release] = pd.DataFrame(\n            {\n                'Sample': pd.Series(samples),\n                'Variants called': pd.Series(num_called),\n                'Variants missing': pd.Series(num_called),\n                'Proportion missing': pd.Series(pc_missing),\n                'Proportion pass missing': pd.Series(pc_pass_missing),\n                'Proportion heterozygous': pd.Series(pc_het),\n                'Proportion pass heterozygous': pd.Series(pc_pass_het),\n                'Proportion homozygous alternative': pd.Series(pc_hom_alt),\n                'Proportion pass homozygous alternative': pd.Series(pc_pass_hom_alt),\n                'Proportion variants SNPs': pd.Series(pc_snp),\n                'Proportion indels insertions': pd.Series(pc_ins),\n            }\n        )")

is_ins

num_snp_hom_ref = genotypes_subset['Pv30']['pass'][is_snp, :, :].count_hom_ref(axis=0)[:]

genotypes_subset['Pv30']['pass'].subset(is_snp)

pd.value_counts(is_snp)

len(is_snp)

is_snp = (hdf[release]['variants']['VARIANT_TYPE'][:][pass_variants] == b'SNP')

2+2

df_sample_summary['Pv30']

df_sample_summary['Pf60']



