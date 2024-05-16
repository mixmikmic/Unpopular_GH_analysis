get_ipython().run_line_magic('run', '_standard_imports.ipynb')

panoptes_metadata_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/pf_6_0_panoptes_samples_20170124.txt.gz"
panoptes_final_metadata_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/pf_6_0_panoptes_samples_final_20170124.txt.gz"
oxford_table_fn = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161214_samples_which_release/oxford.txt"
pf3k_metadata_fn = "/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/pf3k_release_5_metadata.txt"
crosses_metadata_fn = "/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/pf3k_release_5_crosses_metadata.txt"
new_pf3k_metadata_fn = "/nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/pf3k_release_5_metadata_20170210.txt.gz"

tbl_pf_60_metadata = (
    etl
    .fromtsv(panoptes_metadata_fn)
#     .cut(['Sample', 'pf60_OxfordSrcCode', 'pf60_OxfordDonorCode', 'Individual_ID', 'pf60_AlfrescoStudyCode',
#           'pf60_HasDuplicate', 'pf60_DiscardAsDuplicate', 'pc_pass_missing', 'pc_genome_covered_at_1x'])
)
print(len(tbl_pf_60_metadata.data()))
tbl_pf_60_metadata

(
    tbl_pf_60_metadata
    .valuecounts('AlfrescoStudyCode')
    .displayall()
)

(
    tbl_pf_60_metadata
    .selectin('AlfrescoStudyCode', ['1155-PF-ID-PRICE', '1106-PV-MULTI-PRICE'])
    .valuecounts('pf60_Location')
    .displayall()
)

tbl_pf_60_metadata = (
    etl
    .fromtsv(panoptes_final_metadata_fn)
#     .cut(['Sample', 'OxfordSrcCode', 'OxfordDonorCode', 'Individual_ID', 'IndividualGroup', 'AlfrescoStudyCode',
#           'HasDuplicate', 'DiscardAsDuplicate', 'pc_pass_missing', 'pc_genome_covered_at_1x'])
    .cut(['Sample', 'OxfordSrcCode', 'OxfordDonorCode', 'Individual_ID'])
)
print(len(tbl_pf_60_metadata.data()))
tbl_pf_60_metadata

tbl_pf3k_metadata = (
    etl
    .fromtsv(pf3k_metadata_fn)
)
print(len(tbl_pf3k_metadata.data()))
tbl_pf3k_metadata

tbl_oxford = (
    etl
    .fromtsv(oxford_table_fn)
    .distinct('oxford_code')
    .cut(['oxford_code', 'oxford_source_code', 'oxford_donor_source_code'])
)
print(len(tbl_oxford.distinct('oxford_code').data()))
print(len(tbl_oxford.data()))
tbl_oxford

tbl_crosses = (
    etl
    .fromtsv(crosses_metadata_fn)
    .cut(['sample', 'study', 'clone'])
)
print(len(tbl_crosses.distinct('sample').data()))
print(len(tbl_crosses.data()))
tbl_crosses

def is_duplicate(prv, cur, nxt):
    if prv is None and (nxt['Individual_ID'] == cur['Individual_ID']):
        return(True)
    elif prv is None and (nxt['Individual_ID'] != cur['Individual_ID']):
        return(False)
    elif nxt is None and (prv['Individual_ID'] == cur['Individual_ID']):
        return(True)
    elif nxt is None and (prv['Individual_ID'] != cur['Individual_ID']):
        return(False)
    elif(
            (prv['Individual_ID'] == cur['Individual_ID']) or
            (nxt['Individual_ID'] == cur['Individual_ID'])
        ):
        return(True)
    else:
        return(False)
    
def discard_as_duplicate(prv, cur, nxt):
    if prv is None:
        return(False)
    elif (prv['Individual_ID'] == cur['Individual_ID']):
        return(True)
    else:
        return(False)
    

# final_columns = list(tbl_pf3k_metadata.header()) + ['IsFieldSample', 'PreferredSample', 'AllSamplesThisIndividual', 'DiscardAsDuplicate', 'HasDuplicate', 'Individual_ID']
final_columns = list(tbl_pf3k_metadata.header()) + ['IsFieldSample', 'PreferredSample', 'AllSamplesThisIndividual']
final_columns

tbl_temp = (
    tbl_pf3k_metadata
    .leftjoin(tbl_pf_60_metadata, lkey='sample', rkey='Sample')
    .leftjoin(tbl_oxford, lkey='sample', rkey='oxford_code')
    .leftjoin(tbl_crosses, key='sample', rprefix='crosses_')
    .convert('Individual_ID', lambda v, r: r['crosses_clone'], where=lambda r: r['study'] in ['1041', '1042', '1043'], pass_row=True)
    .convert('Individual_ID', lambda v, r: r['sample'], where=lambda r: r['study'] in ['Broad Senegal', '1104', ''], pass_row=True)
    .convert('Individual_ID', lambda v, r: r['oxford_donor_source_code'], where=lambda r: r['Individual_ID'] is None, pass_row=True)
    .convert('Individual_ID', lambda v, r: 'PF955', where=lambda r: r['Individual_ID'] == 'PF955_MACS', pass_row=True)
    .addfield('IsFieldSample', lambda r: r['country'] != '')
    .sort(['Individual_ID', 'bases_of_5X_coverage'], reverse=True)
    .addfieldusingcontext('HasDuplicate', is_duplicate)
    .addfieldusingcontext('DiscardAsDuplicate', discard_as_duplicate)
    .addfield('PreferredSample', lambda rec: rec['DiscardAsDuplicate'] == False)
)

# tbl_temp.totsv('temp.txt')
# tbl_temp = etl.fromtsv('temp.txt')

tbl_duplicates_sample = (
    tbl_temp
    .aggregate('Individual_ID', etl.strjoin(','), 'sample')
    .rename('value', 'AllSamplesThisIndividual')
)

tbl_new_pf3k_metadata = (
    tbl_temp
    .leftjoin(tbl_duplicates_sample, key='Individual_ID')
    .cut(final_columns)
    .sort('sample')
    .sort('IsFieldSample', reverse=True)
)

print(len(tbl_new_pf3k_metadata.distinct('sample').data()))
print(len(tbl_new_pf3k_metadata.data()))
tbl_new_pf3k_metadata

tbl_new_pf3k_metadata.totsv(new_pf3k_metadata_fn, lineterminator='\n')
# tbl_new_pf3k_metadata.totsv(new_pf3k_metadata_fn)

# Sanity check new file is same as old but with extra_columns
get_ipython().system('zcat {new_pf3k_metadata_fn} | cut -f 1-23 > /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/temp_remove_1.txt')

get_ipython().system('sed \'s/"//g\' /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/pf3k_release_5_metadata.txt > /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/temp_remove_2.txt')

get_ipython().system('diff /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/temp_remove_1.txt /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/temp_remove_2.txt')

get_ipython().system('rm /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/temp_remove_1.txt')
get_ipython().system('rm /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0/temp_remove_2.txt')



tbl_new_pf3k_metadata.selecteq('Individual_ID', 'PF955')

tbl_new_pf3k_metadata.valuecounts('IsFieldSample')

tbl_new_pf3k_metadata.valuecounts('PreferredSample')

tbl_new_pf3k_metadata.valuecounts('DiscardAsDuplicate')

tbl_new_pf3k_metadata.selecteq('PreferredSample', False).displayall()

tbl_new_pf3k_metadata.selecteq('Individual_ID', 'EIMK002').displayall()

tbl_temp.valuecounts('DiscardAsDuplicate')

tbl_new_pf3k_metadata.valuecounts('HasDuplicate')

tbl_new_pf3k_metadata.selectnone('HasDuplicate').valuecounts('study').displayall()

tbl_new_pf3k_metadata.selectnone('HasDuplicate')

tbl_new_pf3k_metadata.selecteq('Individual_ID', 'PFD140')

tbl_new_pf3k_metadata.selecteq('Individual_ID', '')

tbl_new_pf3k_metadata.selectnone('Individual_ID')

len(tbl_new_pf3k_metadata.selecteq('study', 'Broad Senegal').data())

print(len(tbl_new_pf3k_metadata.distinct('AllSamplesThisIndividual').data()))
print(len(tbl_new_pf3k_metadata.distinct(('study', 'AllSamplesThisIndividual')).data()))





tbl_new_pf3k_metadata.selecteq('Individual_ID', 'HB3')

tbl_new_pf3k_metadata.selecteq('Individual_ID', 'C02')

tbl_new_pf3k_metadata.selecteq('Individual_ID', 'A4')





tbl_new_pf3k_metadata.valuecounts('study').displayall()

(
    tbl_new_pf3k_metadata
    .selectnone('Individual_ID')
    .valuecounts('study')
    .displayall()
)

(
    tbl_new_pf3k_metadata
    .selectnone('Individual_ID')
    .valuecounts('study')
    .displayall()
)

(
    tbl_new_pf3k_metadata
    .selecteq('Individual_ID', '')
    .valuecounts('study')
    .displayall()
)

(
    tbl_new_pf3k_metadata
    .selectnotin('study', ['Broad Senegal', '', '1041', '1042', '1043', '1104'])
    .selecteq('AlfrescoStudyCode', None)
    .valuecounts('study', 'AlfrescoStudyCode')
    .displayall()
)

(
    tbl_new_pf3k_metadata
    .selectnotin('study', ['Broad Senegal', '', '1041', '1042', '1043', '1104'])
    .selecteq('AlfrescoStudyCode', None)
    .displayall()
)

(
    tbl_new_pf3k_metadata
    .selectnone('Individual_ID')
    .selecteq('AlfrescoStudyCode', None)
    .displayall()
)



