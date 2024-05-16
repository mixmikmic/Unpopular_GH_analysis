import gcp.bigquery as bq
cn_BQtable = bq.Table('isb-cgc:tcga_201607_beta.Copy_Number_segments')

get_ipython().magic('bigquery schema --table $cn_BQtable')

get_ipython().run_cell_magic('sql', '--module count_unique', '\nDEFINE QUERY q1\nSELECT COUNT (DISTINCT $f, 25000) AS n\nFROM $t')

fieldList = ['ParticipantBarcode', 'SampleBarcode', 'AliquotBarcode']
for aField in fieldList:
  field = cn_BQtable.schema[aField]
  rdf = bq.Query(count_unique.q1,t=cn_BQtable,f=field).results().to_dataframe()
  print " There are %6d unique values in the field %s. " % ( rdf.iloc[0]['n'], aField)

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  SampleTypeLetterCode,\n  COUNT(*) AS n\nFROM (\n  SELECT\n    SampleTypeLetterCode,\n    SampleBarcode\n  FROM\n    $cn_BQtable\n  GROUP BY\n    SampleTypeLetterCode,\n    SampleBarcode )\nGROUP BY\n  SampleTypeLetterCode\nORDER BY\n  n DESC')

get_ipython().run_cell_magic('sql', '', '\nSELECT\n  MIN(Length) AS minLength,\n  MAX(Length) AS maxLength,\n  AVG(Length) AS avgLength,\n  STDDEV(Length) AS stdLength,\n  MIN(Num_Probes) AS minNumProbes,\n  MAX(Num_Probes) AS maxNumProbes,\n  AVG(Num_Probes) AS avgNumProbes,\n  STDDEV(Num_Probes) AS stdNumProbes,\n  MIN(Segment_Mean) AS minCN,\n  MAX(Segment_Mean) AS maxCN,\n  AVG(Segment_Mean) AS avgCN,\n  STDDEV(Segment_Mean) AS stdCN,\nFROM (\n  SELECT\n    Start,\n    END,\n    (End-Start+1) AS Length,\n    Num_Probes,\n    Segment_Mean\n  FROM\n    $cn_BQtable )')

import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_cell_magic('sql', '--module getCNhist', '\nSELECT\n  lin_bin,\n  COUNT(*) AS n\nFROM (\n  SELECT\n    Segment_Mean,\n    (2.*POW(2,Segment_Mean)) AS lin_CN,\n    INTEGER(((2.*POW(2,Segment_Mean))+0.50)/1.0) AS lin_bin\n  FROM\n    $t\n  WHERE\n    ( (End-Start+1)>1000 AND SampleTypeLetterCode="TP" ) )\nGROUP BY\n  lin_bin\nHAVING\n  ( n > 2000 )\nORDER BY\n  lin_bin ASC')

CNhist = bq.Query(getCNhist,t=cn_BQtable).results().to_dataframe()
bar_width=0.80
plt.bar(CNhist['lin_bin']+0.1,CNhist['n'],bar_width,alpha=0.8);
plt.xticks(CNhist['lin_bin']+0.5,CNhist['lin_bin']);
plt.title('Histogram of Average Copy-Number');
plt.ylabel('# of segments');
plt.xlabel('integer copy-number');

get_ipython().run_cell_magic('sql', '--module getSLhist_1k', '\nSELECT\n  bin,\n  COUNT(*) AS n\nFROM (\n  SELECT\n    (END-Start+1) AS segLength,\n    INTEGER((END-Start+1)/1000) AS bin\n  FROM\n    $t\n  WHERE\n    (END-Start+1)<1000000 AND SampleTypeLetterCode="TP" )\nGROUP BY\n  bin\nORDER BY\n  bin ASC')

SLhist_1k = bq.Query(getSLhist_1k,t=cn_BQtable).results().to_dataframe()
plt.plot(SLhist_1k['bin'],SLhist_1k['n'],'ro:');
plt.xscale('log');
plt.yscale('log');
plt.xlabel('Segment length (Kb)');
plt.ylabel('# of Segments');
plt.title('Distribution of Segment Lengths');

get_ipython().run_cell_magic('sql', '--module getSLhist', '\nSELECT\n  bin,\n  COUNT(*) AS n\nFROM (\n  SELECT\n    (END-Start+1) AS segLength,\n    INTEGER((END-Start+1)/1) AS bin\n  FROM\n    $t\n  WHERE\n    (END-Start+1)<1000 AND SampleTypeLetterCode="TP" )\nGROUP BY\n  bin\nORDER BY\n  bin ASC')

SLhist = bq.Query(getSLhist,t=cn_BQtable).results().to_dataframe()
plt.plot(SLhist['bin'],SLhist['n'],'ro:');
plt.xscale('log');
plt.yscale('log');
plt.xlabel('Segment length (bp)');
plt.ylabel('# of Segments');
plt.title('Distribution of Segment Lengths (<1Kb)');

get_ipython().run_cell_magic('sql', '--module getSLhist_1k_del', '\nSELECT\n  bin,\n  COUNT(*) AS n\nFROM (\n  SELECT\n    (END-Start+1) AS segLength,\n    INTEGER((END-Start+1)/1000) AS bin\n  FROM\n    $t\n  WHERE\n    (END-Start+1)<1000000 AND SampleTypeLetterCode="TP" AND Segment_Mean<-0.7 )\nGROUP BY\n  bin\nORDER BY\n  bin ASC')

get_ipython().run_cell_magic('sql', '--module getSLhist_1k_amp', '\nSELECT\n  bin,\n  COUNT(*) AS n\nFROM (\n  SELECT\n    (END-Start+1) AS segLength,\n    INTEGER((END-Start+1)/1000) AS bin\n  FROM\n    $t\n  WHERE\n    (END-Start+1)<1000000 AND SampleTypeLetterCode="TP" AND Segment_Mean>0.7 )\nGROUP BY\n  bin\nORDER BY\n  bin ASC')

SLhistDel = bq.Query(getSLhist_1k_del,t=cn_BQtable).results().to_dataframe()
SLhistAmp = bq.Query(getSLhist_1k_amp,t=cn_BQtable).results().to_dataframe()

plt.plot(SLhist_1k['bin'],SLhist_1k['n'],'ro:');
plt.plot(SLhistDel['bin'],SLhistDel['n'],'bo-')
plt.plot(SLhistAmp['bin'],SLhistDel['n'],'go-',alpha=0.3)
plt.xscale('log');
plt.yscale('log');
plt.xlabel('Segment length (Kb)');
plt.ylabel('# of Segments');
plt.title('Distribution of Segment Lengths');

get_ipython().run_cell_magic('sql', '--module getGeneCN', '\nSELECT\n  SampleBarcode, \n  AVG(Segment_Mean) AS avgCN,\n  MIN(Segment_Mean) AS minCN,\n  MAX(Segment_Mean) AS maxCN,\nFROM\n  $t\nWHERE\n  ( SampleTypeLetterCode=$sampleType\n    AND Num_Probes > 10\n    AND Chromosome=$geneChr\n    AND ( (Start<$geneStart AND End>$geneStop)\n       OR (Start<$geneStop  AND End>$geneStop)\n       OR (Start>$geneStart AND End<$geneStop) ) )\nGROUP BY\n  SampleBarcode')

# EGFR gene coordinates  
geneChr = "7"
geneStart = 55086725
geneStop = 55275031
egfrCN = bq.Query(getGeneCN,t=cn_BQtable,sampleType="TP",geneChr=geneChr,geneStart=geneStart,geneStop=geneStop).results().to_dataframe()

# MYC gene coordinates
geneChr = "8"
geneStart = 128748315
geneStop = 128753680
mycCN = bq.Query(getGeneCN,t=cn_BQtable,sampleType="TP",geneChr=geneChr,geneStart=geneStart,geneStop=geneStop).results().to_dataframe()

# TP53 gene coordinates
geneChr = "17"
geneStart = 7571720
geneStop = 7590868
tp53CN = bq.Query(getGeneCN,t=cn_BQtable,sampleType="TP",geneChr=geneChr,geneStart=geneStart,geneStop=geneStop).results().to_dataframe()

binWidth = 0.2
binVals = np.arange(-2+(binWidth/2.), 6-(binWidth/2.), binWidth)
plt.hist(tp53CN['avgCN'],bins=binVals,normed=False,color='green',alpha=0.9,label='TP53');
plt.hist(mycCN ['avgCN'],bins=binVals,normed=False,color='blue',alpha=0.7,label='MYC');
plt.hist(egfrCN['avgCN'],bins=binVals,normed=False,color='red',alpha=0.5,label='EGFR');
plt.legend(loc='upper right');

plt.hist(tp53CN['avgCN'],bins=binVals,normed=False,color='green',alpha=0.9,label='TP53');
plt.hist(mycCN ['avgCN'],bins=binVals,normed=False,color='blue',alpha=0.7,label='MYC');
plt.hist(egfrCN['avgCN'],bins=binVals,normed=False,color='red',alpha=0.5,label='EGFR');
plt.yscale('log');
plt.legend(loc='upper right');



