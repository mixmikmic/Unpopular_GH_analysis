import gcp.bigquery as bq
mRNAseq_BQtable = bq.Table('isb-cgc:tcga_201607_beta.mRNA_UNC_HiSeq_RSEM')

get_ipython().magic('bigquery schema --table $mRNAseq_BQtable')

get_ipython().run_cell_magic('sql', '--module count_unique', '\nDEFINE QUERY q1\nSELECT COUNT (DISTINCT $f, 25000) AS n\nFROM $t')

fieldList = ['ParticipantBarcode', 'SampleBarcode', 'AliquotBarcode']
for aField in fieldList:
  field = mRNAseq_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=mRNAseq_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)

fieldList = ['original_gene_symbol', 'HGNC_gene_symbol', 'gene_id']
for aField in fieldList:
  field = mRNAseq_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=mRNAseq_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  HGNC_gene_symbol,\n  original_gene_symbol,\n  gene_id\nFROM\n  $mRNAseq_BQtable\nWHERE\n  ( original_gene_symbol IS NOT NULL\n    AND HGNC_gene_symbol IS NOT NULL\n    AND original_gene_symbol=HGNC_gene_symbol\n    AND gene_id IS NOT NULL )\nGROUP BY\n  original_gene_symbol,\n  HGNC_gene_symbol,\n  gene_id\nORDER BY\n  HGNC_gene_symbol')

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  HGNC_gene_symbol,\n  original_gene_symbol,\n  gene_id\nFROM\n  $mRNAseq_BQtable\nWHERE\n  ( original_gene_symbol IS NOT NULL\n    AND HGNC_gene_symbol IS NOT NULL\n    AND original_gene_symbol!=HGNC_gene_symbol\n    AND gene_id IS NOT NULL )\nGROUP BY\n  original_gene_symbol,\n  HGNC_gene_symbol,\n  gene_id\nORDER BY\n  HGNC_gene_symbol')

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  Study,\n  n,\n  exp_mean,\n  exp_sigma,\n  (exp_sigma/exp_mean) AS exp_cv\nFROM (\n  SELECT\n    Study,\n    AVG(LOG2(normalized_count+1)) AS exp_mean,\n    STDDEV_POP(LOG2(normalized_count+1)) AS exp_sigma,\n    COUNT(AliquotBarcode) AS n\n  FROM\n    $mRNAseq_BQtable\n  WHERE\n    ( SampleTypeLetterCode="TP"\n      AND HGNC_gene_symbol="EGFR" )\n  GROUP BY\n    Study )\nORDER BY\n  exp_sigma DESC')

get_ipython().run_cell_magic('sql', '--module highVar', '\nSELECT\n  Study,\n  HGNC_gene_symbol,\n  n,\n  exp_mean,\n  exp_sigma,\n  (exp_sigma/exp_mean) AS exp_cv\nFROM (\n  SELECT\n    Study,\n    HGNC_gene_symbol,\n    AVG(LOG2(normalized_count+1)) AS exp_mean,\n    STDDEV_POP(LOG2(normalized_count+1)) AS exp_sigma,\n    COUNT(AliquotBarcode) AS n\n  FROM\n    $t\n  WHERE\n    ( SampleTypeLetterCode="TP" )\n  GROUP BY\n    Study,\n    HGNC_gene_symbol )\nORDER BY\n  exp_sigma DESC')

q = bq.Query(highVar,t=mRNAseq_BQtable)
print q.sql

r = bq.Query(highVar,t=mRNAseq_BQtable).results()

#r.to_dataframe()

get_ipython().run_cell_magic('sql', '--module hv_genes', '\nSELECT *\nFROM ( $hv_result )\nHAVING\n  ( exp_mean > 6.\n    AND n >= 200\n    AND exp_cv > 0.5 )\nORDER BY\n  exp_cv DESC')

bq.Query(hv_genes,hv_result=r).results().to_dataframe()



