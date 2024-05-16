get_ipython().run_line_magic('run', '_standard_imports.ipynb')

output_dir = '/nfs/team112_internal/rp7/data/Pf/hrp'
get_ipython().system('mkdir -p {output_dir}/fofns')
get_ipython().system('mkdir -p {output_dir}/metadata')
cinzia_metadata_fn = '%s/metadata/PF_metadata_base.csv' % output_dir # From Cinzia 22/03/2016 07:47
v4_metadata_fn = '%s/metadata/PGV4_mk5.xlsx' % output_dir # From Roberto 14/06/2016 15:17
v5_metadata_fn = '%s/metadata/v5.xlsx' % output_dir # From Roberto 14/06/2016 15:17
iso_country_code_fn = '%s/metadata/country-codes.csv' % output_dir # https://raw.githubusercontent.com/datasets/country-codes/master/data/country-codes.csv
regions_in_dataset_fn = '%s/metadata/regions_in_dataset.xlsx' % output_dir
sub_contintent_fn = '%s/metadata/region_sub_continents.xlsx' % output_dir
manifest_fn = '%s/metadata/hrp_manifest_20160620.txt' % output_dir
jim_manifest_fn = '%s/metadata/manifest_for_jim_20160620.txt' % output_dir
lookseq_fn = '%s/metadata/lookseq.txt' % output_dir

lab_studies = list(range(1032, 1044, 1)) + [1104, 1133, 1150, 1153]

# cinzia_extra_metadata_fn = '/nfs/team112_internal/rp7/data/Pf/4_0/meta/PF_extrametadata.csv' # From Cinzia 22/03/2016 08:22

fofns = collections.OrderedDict()

fofns['pf_community_5_0'] = '/nfs/team112_internal/production/release_build/Pf/5_0_release_packages/pf_50_freeze_manifest_nolab_olivo.tab'
fofns['pf_community_5_1'] = '/nfs/team112_internal/production_files/Pf/5_1/pf_51_samplebam_cleaned.fofn'
fofns['pf3k_pilot_5_0_broad'] = '/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/pf3k_metadata.tab'
fofns['pdna'] = '/nfs/team112_internal/production_files/Pf/PDNA/pf_pdna_new_samplebam.fofn'
fofns['conway'] = '/nfs/team112_internal/production_files/Pf/1147_Conway/pf_conway_metadata.fofn'
fofns['trac'] = '%s/fofns/olivo_TRAC.fofn' % output_dir
fofns['fanello'] = '%s/fofns/olivo_fanello.fofn' % output_dir

import glob

bam_dirs = collections.OrderedDict()
bam_dirs['trac'] = '/nfs/team112_internal/production_files/Pf/olivo_TRAC/remapped'
bam_dirs['fanello'] = '/nfs/team112_internal/production_files/Pf/olivo_fanello'

for bam_dir in bam_dirs:
    get_ipython().system('rm {fofns[bam_dir]}')
    with open(fofns[bam_dir], "a") as fofn:
# glob.glob('%s/*.bam' % fofns['trac'])
        print("path\tsample", file=fofn)
        for x in glob.glob('%s/*.bam' % bam_dirs[bam_dir]):
            print("%s\t%s" % (x, os.path.basename(x).replace('_', '-').replace('.bam', '')), file=fofn)
# [os.path.basename(x) for x in glob.glob('%s/*.bam' % fofns['trac'])]

for i, fofn in enumerate(fofns):
    if i == 0:
        tbl_all_bams = etl.fromtsv(fofns[fofn]).cut(['path', 'sample', 'bases_of_5X_coverage', 'mean_coverage']).addfield('dataset', fofn)
    else:
        if fofn == 'pf3k_pilot_5_0_broad':
            tbl_all_bams = (
                tbl_all_bams
                .cat(
                    etl.fromtsv(fofns[fofn])
                    .selecteq('study', 'Pf3k_Senegal')
                    .cut(['path', 'sample', 'bases_of_5X_coverage', 'mean_coverage'])
                    .addfield('dataset', fofn)
                )
            )
        elif fofn in ['pf_community_5_0', 'conway']:
            tbl_all_bams = (
                tbl_all_bams
                .cat(
                    etl.fromtsv(fofns[fofn])
                    .cut(['path', 'sample', 'bases_of_5X_coverage', 'mean_coverage'])
                    .addfield('dataset', fofn)
                )
            )
        else:
            tbl_all_bams = (
                tbl_all_bams
                .cat(
                    etl.fromtsv(fofns[fofn])
                    .cut(['path', 'sample'])
                    .addfield('dataset', fofn)
                )
            )
        

len(tbl_all_bams.data())

len(tbl_all_bams.duplicates('sample').data())

len(tbl_all_bams.unique('sample').data())

tbl_all_bams

tbl_all_bams.valuecounts('dataset').displayall()

tbl_solaris = (
    etl.fromcsv(cinzia_metadata_fn, encoding='latin1')
    .cut(['oxford_code', 'type', 'country', 'country_code', 'oxford_src_code', 'oxford_donor_code', 'alfresco_code'])
    .convert('alfresco_code', int)
    .rename('country', 'solaris_country')
    .rename('country_code', 'solaris_country_code')
    .distinct('oxford_code')
)
tbl_solaris.selectne('alfresco_code', None)

tbl_solaris.duplicates('oxford_code').displayall()

tbl_v4_metadata = etl.fromxlsx(v4_metadata_fn, 'PGV4.0').cut(['Sample', 'Region']).rename('Region', 'v4_region')
tbl_v5_metadata = etl.fromxlsx(v5_metadata_fn).cut(['Sample', 'Region']).rename('Region', 'v5_region')

tbl_v4_metadata

tbl_v4_metadata.selecteq('Sample', 'PF0542-C')

tbl_v5_metadata

tbl_v5_metadata.selecteq('Sample', 'PF0542-C')

def determine_region(rec, null_vals=(None, 'NULL', '-')):
#     if (
#         rec['sample'].startswith('PG') or
#         rec['sample'].startswith('PL') or
#         rec['sample'].startswith('PF') or
#         rec['sample'].startswith('WL') or
#         rec['sample'].startswith('WH') or
#         rec['sample'].startswith('WS')
#     ):
    if rec['alfresco_code'] in lab_studies:
        return('Lab')
    if rec['v5_region'] not in null_vals:
        return(rec['v5_region'])
    elif rec['v4_region'] not in null_vals:
        return(rec['v4_region'])
    elif rec['sample'].startswith('PJ'):
        return('ID')
#     elif rec['sample'].startswith('QM'):
#         return('MG')
#     elif rec['sample'].startswith('QS'):
#         return('MG')
    elif rec['solaris_country_code'] not in null_vals:
        return(rec['solaris_country_code'])
    elif rec['solaris_country'] not in null_vals:
        return(rec['solaris_country'])
    else:
        return('unknown')
       

tbl_regions = (
    etl.fromxlsx(v4_metadata_fn, 'Locations')
    .cut(['Country', 'Region'])
    .rename('Country', 'country_from_region')
    .rename('Region', 'region')
    .selectne('region', '-')
    .distinct(['country_from_region', 'region'])
)
tbl_regions.selecteq('country_from_region', 'KH')

tbl_country_code = (
    etl.fromxlsx(v4_metadata_fn, 'CountryCodes')
    .rename('County', 'country')
    .rename('Code', 'code_from_country')
    .rename('SubContinent', 'subcontintent')
    .selectne('country', '-')
)
tbl_country_code.displayall()

def determine_region_code(rec, null_vals=(None, 'NULL', '-')):
    if rec['code_from_country'] not in null_vals:
        return(rec['code_from_country'])
    elif rec['dataset'] == 'pf3k_pilot_5_0_broad':
        return('SN')
    else:
        return(rec['region'])

def determine_country_code(rec, null_vals=(None, 'NULL', '-')):
    if rec['country_from_region'] not in null_vals:
        return(rec['country_from_region'])
    else:
        return(rec['region_code'])

tbl_iso_country_codes = (
    etl.fromcsv(iso_country_code_fn)
    .cut(['official_name', 'ISO3166-1-Alpha-2'])
    .rename('official_name', 'country_name')
)
tbl_iso_country_codes

tbl_sub_continents = etl.fromxlsx(sub_contintent_fn)
tbl_sub_continents

tbl_sub_continent_names = etl.fromxlsx(sub_contintent_fn, 'Names').convertnumbers()
tbl_sub_continent_names

final_fields = [
    'path', 'sample', 'oxford_src_code', 'oxford_donor_code', 'dataset', 'type', 'region_code', 'country_code',
    'country_name', 'sub_continent', 'sub_continent_name', 'sub_continent_number', 'bases_of_5X_coverage', 'mean_coverage'
]

tbl_manifest = (
    tbl_all_bams
    .leftjoin(tbl_solaris, lkey='sample', rkey='oxford_code')
    .replace('type', None, 'unknown')
    .leftjoin(tbl_v4_metadata, lkey='sample', rkey='Sample')
    .leftjoin(tbl_v5_metadata, lkey='sample', rkey='Sample')
    .addfield('region', determine_region)
    .leftjoin(tbl_country_code.cut(['country', 'code_from_country']), lkey='region', rkey='country')
    .addfield('region_code', determine_region_code)
    .replace('region_code', 'Benin', 'BJ')
    .replace('region_code', 'Mauritania', 'MR')
    .replace('region_code', "Cote d'Ivoire (Ivory Coast)", 'CI')
    .replace('region_code', 'Ethiopia', 'ET')
    .replace('region_code', 'US', 'Lab')
    .leftjoin(tbl_regions, lkey='region_code', rkey='region')
    .addfield('country_code', determine_country_code)
    .leftjoin(tbl_iso_country_codes, lkey='country_code', rkey='ISO3166-1-Alpha-2')
    .replace('country_name', "Côte d'Ivoire", "Ivory Coast")
    .leftjoin(tbl_sub_continents.cut(['region_code', 'sub_continent']), key='region_code')
    .leftjoin(tbl_sub_continent_names, key='sub_continent')
    .selectne('region_code', 'unknown')
    .sort(['sub_continent_number', 'country_name', 'sample'])
    .cut(final_fields)
)

tbl_manifest.selecteq('country_code', 'CI')

tbl_manifest.valuecounts('sub_continent').displayall()

tbl_manifest.valuecounts('sub_continent').displayall()

tbl_manifest.selectnone('sub_continent').displayall()

tbl_manifest.selecteq('sub_continent', 'Lab')

len(tbl_all_bams.leftjoin(tbl_solaris, lkey='sample', rkey='oxford_code').data())

manifest_fn

tbl_manifest.totsv(manifest_fn)

tbl_manifest.selectne('dataset', 'pf3k_pilot_5_0_broad').cut(['path', 'sample']).totsv(jim_manifest_fn)

manifest_fn

len(tbl_manifest.data())

len(tbl_manifest.selectne('dataset', 'pf3k_pilot_5_0_broad').cut(['path', 'sample']).data())

len(tbl_manifest.distinct('sample').data())

tbl_temp = tbl_manifest.addfield('bam_exists', lambda rec: os.path.exists(rec['path']))
tbl_temp.valuecounts('bam_exists')

with open(lookseq_fn, "w") as fo:
    for rec in tbl_manifest:
        bam_fn = rec[0]
        sample_name = "%s_%s" % (rec[1].replace('-', '_'), rec[3])
        group_name = "%s %s %s %s" % (rec[9], rec[7], rec[6], rec[4])
        print(
            '"%s" : { "bam":"%s", "species" : "pf_3d7_v3" , "alg" : "bwa" , "group" : "%s" } ,' % (sample_name, bam_fn, group_name),
            file=fo
        )
# "PG0049_CW2" : { "bam":"/lustre/scratch109/malaria/pfalciparum/output/e/b/8/6/144292/1_bam_merge/pe.1.bam" , "species" : "pf_3d7_v3" , "alg" : "bwa" , "group" : "1104-PF-LAB-WENDLER" } ,

2+2

tbl_manifest

tbl_manifest.tail(5)

tbl_manifest.select(lambda rec: rec['sample'].startswith('QZ')).displayall()

tbl_manifest.selecteq('sample', 'WL0071-C').displayall()

tbl_manifest.selecteq('country_code', 'UK').displayall()

tbl_manifest.valuecounts('type', 'dataset').displayall()

lkp_country_code = etl.lookup(tbl_country_code, 'code', 'country')

tbl_manifest.selecteq('type', 'unknown').selecteq('dataset', 'pf_community_5_1')

tbl_manifest.select(lambda rec: rec['v4_region'] is not None and rec['v5_region'] is not None and rec['v4_region'] != rec['v5_region'])

tbl_manifest.select(lambda rec: rec['v4_region'] is not None and rec['v5_region'] is not None)

tbl_manifest.select(lambda rec: rec['v4_region'] is None and rec['v5_region'] is not None)

tbl_manifest.select(lambda rec: rec['v4_region'] is not None and rec['v5_region'] is None)

lkp_code_country = etl.lookup(tbl_country_code, 'country', 'code')

lkp_country_code['BZ']

tbl_manifest.valuecounts('region').displayall()

tbl_manifest.selecteq('region', 'unknown').valuecounts('dataset').displayall()

tbl_manifest.valuecounts('region_code').displayall()

tbl_manifest.selecteq('region_code', 'unknown').valuecounts('dataset').displayall()

tbl_manifest.valuecounts('country_code').displayall()

tbl_manifest.selecteq('country_code', 'unknown').valuecounts('dataset').displayall()

tbl_manifest.valuecounts('country_name').displayall()

tbl_manifest.valuecounts('region_code', 'country_name').toxlsx(regions_in_dataset_fn)

tbl_manifest.valuecounts('sub_continent').displayall()

tbl_manifest.cut(['path', 'sample', 'dataset', 'type', 'region_code', 'country_code', 'country_name', 'sub_continent'])



