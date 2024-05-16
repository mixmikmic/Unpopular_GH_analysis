import gcp.bigquery as bq
annotations_BQtable = bq.Table('isb-cgc:tcga_201607_beta.Annotations')

get_ipython().magic('bigquery schema --table $annotations_BQtable')

get_ipython().run_cell_magic('sql', '', '\nSELECT itemTypeName, COUNT(*) AS n\nFROM $annotations_BQtable\nGROUP BY itemTypeName\nORDER BY n DESC')

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  annotationClassification,\n  annotationCategoryName,\n  COUNT(*) AS n\nFROM\n  $annotations_BQtable\nWHERE\n  ( itemTypeName="Patient" )\nGROUP BY\n  annotationClassification,\n  annotationCategoryName\nHAVING ( n >= 50 )\nORDER BY\n  n DESC')

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  annotationClassification,\n  annotationCategoryName,\n  itemTypeName,\n  COUNT(*) AS n\nFROM\n  $annotations_BQtable\nWHERE\n  ( itemTypeName!="Patient" )\nGROUP BY\n  annotationClassification,\n  annotationCategoryName,\n  itemTypeName\nHAVING ( n >= 50 )\nORDER BY\n  n DESC')

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  Study,\n  COUNT(*) AS n\nFROM\n  $annotations_BQtable\nWHERE\n  ( annotationCategoryName="Item is noncanonical" )\nGROUP BY\n  Study\nORDER BY\n  n DESC')

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  itemTypeName,\n  COUNT(*) AS n\nFROM\n  $annotations_BQtable\nWHERE\n  ( annotationCategoryName="Item is noncanonical"\n    AND Study="OV" )\nGROUP BY\n  itemTypeName\nORDER BY\n  n DESC')

get_ipython().run_cell_magic('sql', '', '\nSELECT\n Study,\n itemTypeName,\n itemBarcode,\n annotationCategoryName,\n annotationClassification,\n ParticipantBarcode,\n SampleBarcode,\n AliquotBarcode,\n LENGTH(itemBarcode) AS n\nFROM\n  $annotations_BQtable\nWHERE\n  ( ParticipantBarcode="TCGA-61-1916" )\nORDER BY n ASC')

get_ipython().run_cell_magic('sql', '', '\nSELECT\n Study,\n itemTypeName,\n itemBarcode,\n annotationCategoryName,\n annotationClassification,\n ParticipantBarcode,\n SampleBarcode,\n AliquotBarcode,\n LENGTH(itemBarcode) AS n\nFROM\n  $annotations_BQtable\nWHERE\n  ( ParticipantBarcode="TCGA-GN-A261" )\nORDER BY n ASC')

get_ipython().run_cell_magic('sql', '', '\nSELECT\n Study,\n itemTypeName,\n itemBarcode,\n annotationCategoryName,\n annotationClassification,\n annotationNoteText,\n ParticipantBarcode,\n SampleBarcode,\n AliquotBarcode,\n LENGTH(itemBarcode) AS n\nFROM\n  $annotations_BQtable\nWHERE\n  ( ParticipantBarcode="TCGA-RS-A6TP" )\nORDER BY n ASC')



