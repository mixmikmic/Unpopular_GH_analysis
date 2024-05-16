import gcp.bigquery as bq
miRNA_BQtable = bq.Table('isb-cgc:tcga_201607_beta.miRNA_Expression')

get_ipython().magic('bigquery schema --table $miRNA_BQtable')

get_ipython().run_cell_magic('sql', '--module count_unique', '\nDEFINE QUERY q1\nSELECT COUNT (DISTINCT $f, 25000) AS n\nFROM $t')

fieldList = ['ParticipantBarcode', 'SampleBarcode', 'AliquotBarcode']
for aField in fieldList:
  field = miRNA_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=miRNA_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)

fieldList = ['mirna_id', 'mirna_accession']
for aField in fieldList:
  field = miRNA_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=miRNA_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  Platform,\n  COUNT(*) AS n\nFROM\n  $miRNA_BQtable\nGROUP BY\n  Platform\nORDER BY\n  n DESC')

