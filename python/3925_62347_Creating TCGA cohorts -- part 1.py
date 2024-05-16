import gcp.bigquery as bq

d = bq.DataSet('isb-cgc:tcga_201607_beta')
for t in d.tables():
  print '%10d rows  %12d bytes   %s'       % (t.metadata.rows, t.metadata.size, t.name.table_id)

get_ipython().magic('bigquery schema --table isb-cgc:tcga_201607_beta.Clinical_data')

table = bq.Table('isb-cgc:tcga_201607_beta.Clinical_data')
if ( table.exists() ):
    fieldNames = map(lambda tsf: tsf.name, table.schema)
    fieldTypes = map(lambda tsf: tsf.data_type, table.schema)
    print " This table has %d fields. " % ( len(fieldNames) )
    print " The first few field names and types are: " 
    print "     ", fieldNames[:5]
    print "     ", fieldTypes[:5]
else: 
    print " There is no existing table called %s:%s.%s" % ( table.name.project_id, table.name.dataset_id, table.name.table_id )

get_ipython().run_cell_magic('sql', '', '\nSELECT tobacco_smoking_history, COUNT(*) AS n\nFROM [isb-cgc:tcga_201607_beta.Clinical_data]\nGROUP BY tobacco_smoking_history\nORDER BY n DESC')

numPatients = table.metadata.rows
print " The %s table describes a total of %d patients. " % ( table.name.table_id, numPatients )

# let's set a threshold for the minimum number of values that a field should have,
# and also the maximum number of unique values
minNumPatients = int(numPatients*0.80)
maxNumValues = 50

numInteresting = 0
iList = []
for iField in range(len(fieldNames)):
  aField = fieldNames[iField]
  aType = fieldTypes[iField]
  try:
    qString = "SELECT {0} FROM [{1}]".format(aField,table)
    query = bq.Query(qString)
    df = query.to_dataframe()
    summary = df[str(aField)].describe()
    if ( aType == "STRING" ):
      topFrac = float(summary['freq'])/float(summary['count'])
      if ( summary['count'] >= minNumPatients ):
        if ( summary['unique'] <= maxNumValues and summary['unique'] > 1 ):
          if ( topFrac < 0.90 ):
            numInteresting += 1
            iList += [aField]
            print "     > %s has %d values with %d unique (%s occurs %d times) "               % (str(aField), summary['count'], summary['unique'], summary['top'], summary['freq'])
    else:
      if ( summary['count'] >= minNumPatients ):
        if ( summary['std'] > 0.1 ):
          numInteresting += 1
          iList += [aField]
          print "     > %s has %d values (mean=%.0f, sigma=%.0f) "             % (str(aField), summary['count'], summary['mean'], summary['std'])
  except:
    pass

print " "
print " Found %d potentially interesting features: " % numInteresting
print "   ", iList

get_ipython().run_cell_magic('sql', '', 'SELECT menopause_status, COUNT(*) AS n\nFROM [isb-cgc:tcga_201607_beta.Clinical_data]\nWHERE menopause_status IS NOT NULL\nGROUP BY menopause_status\nORDER BY n DESC')

get_ipython().run_cell_magic('sql', '', 'SELECT Study, COUNT(*) AS n\nFROM [isb-cgc:tcga_201607_beta.Clinical_data]\nWHERE menopause_status IS NOT NULL\nGROUP BY Study\nORDER BY n DESC')

get_ipython().run_cell_magic('sql', '', 'SELECT hpv_status, hpv_calls, COUNT(*) AS n\nFROM [isb-cgc:tcga_201607_beta.Clinical_data]\nWHERE hpv_status IS NOT NULL\nGROUP BY hpv_status, hpv_calls\nHAVING n > 20\nORDER BY n DESC')

get_ipython().run_cell_magic('sql', '--module createCohort_and_checkAnnotations', '\nDEFINE QUERY select_on_annotations\nSELECT\n  ParticipantBarcode,\n  annotationCategoryName AS categoryName,\n  annotationClassification AS classificationName\nFROM\n  [isb-cgc:tcga_201607_beta.Annotations]\nWHERE\n  ( itemTypeName="Patient"\n    AND (annotationCategoryName="History of unacceptable prior treatment related to a prior/other malignancy"\n      OR annotationClassification="Redaction" ) )\nGROUP BY\n  ParticipantBarcode,\n  categoryName,\n  classificationName\n\nDEFINE QUERY select_on_clinical\nSELECT\n  ParticipantBarcode,\n  vital_status,\n  days_to_last_known_alive,\n  ethnicity,\n  histological_type,\n  menopause_status,\n  race\nFROM\n  [isb-cgc:tcga_201607_beta.Clinical_data]\nWHERE\n  ( Study="BRCA"\n    AND age_at_initial_pathologic_diagnosis<=50\n    AND gender="FEMALE" )\n\nSELECT\n  c.ParticipantBarcode AS ParticipantBarcode\nFROM (\n  SELECT\n    a.categoryName,\n    a.classificationName,\n    a.ParticipantBarcode,\n    c.ParticipantBarcode,\n  FROM ( $select_on_annotations ) AS a\n  OUTER JOIN EACH \n       ( $select_on_clinical ) AS c\n  ON\n    a.ParticipantBarcode = c.ParticipantBarcode\n  WHERE\n    (a.ParticipantBarcode IS NOT NULL\n      OR c.ParticipantBarcode IS NOT NULL)\n  ORDER BY\n    a.classificationName,\n    a.categoryName,\n    a.ParticipantBarcode,\n    c.ParticipantBarcode )\nWHERE\n  ( a.categoryName IS NULL\n    AND a.classificationName IS NULL\n    AND c.ParticipantBarcode IS NOT NULL )\nORDER BY\n  c.ParticipantBarcode')

bq.Query(createCohort_and_checkAnnotations.select_on_annotations).results().to_dataframe()

bq.Query(createCohort_and_checkAnnotations.select_on_clinical).results().to_dataframe()

bq.Query(createCohort_and_checkAnnotations).results().to_dataframe()

q = bq.Query(createCohort_and_checkAnnotations)
q

q.execute_dry_run()



