import gcp.bigquery as bq
rppa_BQtable = bq.Table('isb-cgc:tcga_201607_beta.Protein_RPPA_data')

get_ipython().magic('bigquery schema --table $rppa_BQtable')

get_ipython().run_cell_magic('sql', '--module count_unique', '\nDEFINE QUERY q1\nSELECT COUNT (DISTINCT $f, 25000) AS n\nFROM $t')

fieldList = ['ParticipantBarcode', 'SampleBarcode', 'AliquotBarcode']
for aField in fieldList:
  field = rppa_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=rppa_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)

fieldList = ['Gene_Name', 'Protein_Name', 'Protein_Basename']
for aField in fieldList:
  field = rppa_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=rppa_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  Gene_Name,\n  COUNT(*) AS n\nFROM (\n  SELECT\n    Gene_Name,\n    Protein_Name,\n  FROM\n    $rppa_BQtable\n  GROUP BY\n    Gene_Name,\n    Protein_Name )\nGROUP BY\n  Gene_Name\nHAVING\n  ( n > 1 )\nORDER BY\n  n DESC')

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  Gene_Name,\n  Protein_Name,\n  Phospho,\n  antibodySource,\n  validationStatus\nFROM\n  $rppa_BQtable\nWHERE\n  ( Gene_Name="EIF4EBP1" )\nGROUP BY\n  Gene_Name,\n  Protein_Name,\n  Phospho,\n  antibodySource,\n  validationStatus\nORDER BY\n  Gene_Name,\n  Protein_Name,\n  Phospho,\n  antibodySource,\n  validationStatus')

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  Gene_Name,\n  Protein_Name,\n  Phospho,\n  antibodySource,\n  validationStatus\nFROM\n  $rppa_BQtable\nWHERE\n  ( Gene_Name CONTAINS "AKT" )\nGROUP BY\n  Gene_Name,\n  Protein_Name,\n  Phospho,\n  antibodySource,\n  validationStatus\nORDER BY\n  Gene_Name,\n  Protein_Name,\n  Phospho,\n  antibodySource,\n  validationStatus')

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  SampleBarcode,\n  Study,\n  Gene_Name,\n  Protein_Name,\n  Protein_Expression\nFROM\n  $rppa_BQtable\nWHERE\n  ( Protein_Name="Akt" )\nORDER BY\n  SampleBarcode,\n  Gene_Name\nLIMIT\n  9')



