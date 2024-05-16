import gcp.bigquery as bq
meth_BQtable = bq.Table('isb-cgc:tcga_201607_beta.DNA_Methylation_betas')

get_ipython().magic('bigquery schema --table $meth_BQtable')

get_ipython().run_cell_magic('sql', '--module count_unique', '\nDEFINE QUERY q1\nSELECT COUNT (DISTINCT $f, 500000) AS n\nFROM $t')

fieldList = ['ParticipantBarcode', 'SampleBarcode', 'AliquotBarcode', 'Probe_Id']
for aField in fieldList:
  field = meth_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=meth_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)

methAnnot = bq.Table('isb-cgc:platform_reference.methylation_annotation')

get_ipython().magic('bigquery schema --table $methAnnot')

get_ipython().run_cell_magic('sql', '--module getGeneProbes', '\nSELECT\n  IlmnID, Methyl27_Loci, CHR, MAPINFO\nFROM\n  $t\nWHERE\n  ( CHR=$geneChr\n    AND ( MAPINFO>$geneStart AND MAPINFO<$geneStop ) )\nORDER BY\n  Methyl27_Loci DESC, \n  MAPINFO ASC')

# MLH1 gene coordinates (+/- 2500 bp)
geneChr = "3"
geneStart = 37034841 - 2500
geneStop  = 37092337 + 2500
mlh1Probes = bq.Query(getGeneProbes,t=methAnnot,geneChr=geneChr,geneStart=geneStart,geneStop=geneStop).results()

mlh1Probes

get_ipython().run_cell_magic('sql', '--module getMLH1methStats', '\nSELECT \n  cpg.IlmnID AS Probe_Id,\n  cpg.Methyl27_Loci AS Methyl27_Loci,\n  cpg.CHR AS Chr,\n  cpg.MAPINFO AS Position,\n  data.beta_stdev AS beta_stdev,\n  data.beta_mean AS beta_mean,\n  data.beta_min AS beta_min,\n  data.beta_max AS beta_max\nFROM (\n  SELECT *\n  FROM $mlh1Probes \n) AS cpg\nJOIN (\n  SELECT \n    Probe_Id,\n    STDDEV(beta_value) beta_stdev,\n    AVG(beta_value) beta_mean,\n    MIN(beta_value) beta_min,\n    MAX(beta_value) beta_max\n    FROM $meth_BQtable\n    WHERE ( SampleTypeLetterCode=$sampleType )\n    GROUP BY Probe_Id\n) AS data\nON \n  cpg.IlmnID = data.Probe_Id\nORDER BY\n  Position ASC')

qTP = bq.Query(getMLH1methStats,mlh1Probes=mlh1Probes,meth_BQtable=meth_BQtable,sampleType="TP")
rTP = qTP.results().to_dataframe()
rTP.describe()

qNT = bq.Query(getMLH1methStats,mlh1Probes=mlh1Probes,meth_BQtable=meth_BQtable,sampleType="NT")
rNT = qNT.results().to_dataframe()
rNT.describe()

import numpy as np
import matplotlib.pyplot as plt

bins=range(1,len(rTP)+1)
#print bins
plt.bar(bins,rTP['beta_mean'],color='red',alpha=0.8,label='Primary Tumor');
plt.bar(bins,rNT['beta_mean'],color='blue',alpha=0.4,label='Normal Tissue');
plt.legend(loc='upper left');
plt.title('MLH1 DNA methylation: average');

plt.bar(bins,rTP['beta_stdev'],color='red',alpha=0.8,label='Primary Tumor');
plt.bar(bins,rNT['beta_stdev'],color='blue',alpha=0.4,label='Normal Tissue');
plt.legend(loc='upper right');
plt.title('MLH1 DNA methylation: standard deviation');

