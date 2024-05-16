get_ipython().run_line_magic('run', 'imports.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')

for release in CHROM_VCF_FNS.keys():
    output_dir = '%s/%s/sites/sites_only_vcfs' % (DATA_DIR, release)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for chrom in CHROM_VCF_FNS[release].keys():
        input_vcf_fn = CHROM_VCF_FNS[release][chrom]
        output_vcf_fn = '%s/%s_%s_sites.vcf.gz' % (output_dir, release, chrom)
        if not os.path.exists(output_vcf_fn):
            get_ipython().system('bcftools view --drop-genotypes --output-type z --output-file {output_vcf_fn} {input_vcf_fn}')

for release in CHROM_VCF_FNS.keys():
    output_dir = '%s/%s/sites/sites_only_vcfs' % (DATA_DIR, release)
    input_files = ' '.join(
        ['%s/%s_%s_sites.vcf.gz' % (output_dir, release, chrom) for chrom in CHROM_VCF_FNS[release].keys()]
    )
    output_vcf_fn = '%s/%s_%s_sites.vcf.gz' % (output_dir, release, 'WG')
    if not os.path.exists(output_vcf_fn):
        get_ipython().system('bcftools concat --output-type z --output {output_vcf_fn} {input_files}')
        get_ipython().system('bcftools index --tbi {output_vcf_fn}')

for release in WG_VCF_FNS.keys():
    print(release)
    output_dir = '%s/%s/sites/sites_only_vcfs' % (DATA_DIR, release)
    if release == 'release3':
        vcf_fn = WG_VCF_FNS['release3']
    else:
        vcf_fn = '%s/%s_%s_sites.vcf.gz' % (output_dir, release, 'WG')
    vcfnp.variants(
        vcf_fn,
        dtypes={
            'REF':                      'a10',
            'ALT':                      'a10',
            'RegionType':               'a25',
            'VariantType':              'a40',
            'RU':                       'a40',
            'set':                      'a40',
            'SNPEFF_AMINO_ACID_CHANGE': 'a20',
            'SNPEFF_CODON_CHANGE':      'a20',
            'SNPEFF_EFFECT':            'a33',
            'SNPEFF_EXON_ID':            'a2',
            'SNPEFF_FUNCTIONAL_CLASS':   'a8',
            'SNPEFF_GENE_BIOTYPE':      'a14',
            'SNPEFF_GENE_NAME':         'a20',
            'SNPEFF_IMPACT':             'a8',
            'SNPEFF_TRANSCRIPT_ID':     'a20',
            'culprit':                  'a14',
        },
        arities={
            'ALT':   6,
            'AF':    6,
            'AC':    6,
            'MLEAF': 6,
            'MLEAC': 6,
            'RPA':   7,
            'ANN':   1,
        },
        fills={
            'VQSLOD': np.nan,
            'QD': np.nan,
            'MQ': np.nan,
            'MQRankSum': np.nan,
            'ReadPosRankSum': np.nan,
            'FS': np.nan,
            'SOR': np.nan,
            'DP': np.nan,
        },
        flatten_filter=True,
        verbose=False,
        cache=True,
        cachedir=output_dir
    )



