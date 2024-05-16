import gcp.bigquery as bq

d = bq.DataSet('isb-cgc:tcga_201607_beta')
for t in d.tables():
  print '%10d rows  %12d bytes   %s'       % (t.metadata.rows, t.metadata.size, t.name.table_id)

