get_ipython().run_line_magic('run', '_standard_imports.ipynb')

scratch_dir = "/lustre/scratch109/malaria/rp7/data/methods-dev/builds/Pf6.0/20161115_run_Olivo_GRC"
output_dir = "/nfs/team112_internal/rp7/data/methods-dev/builds/Pf6.0/20161115_run_Olivo_GRC"
get_ipython().system('mkdir -p {scratch_dir}/grc')
get_ipython().system('mkdir -p {scratch_dir}/species')
get_ipython().system('mkdir -p {scratch_dir}/log')
get_ipython().system('mkdir -p {output_dir}/grc')
get_ipython().system('mkdir -p {output_dir}/species')

bam_fn = "%s/pf_60_mergelanes.txt" % output_dir
bam_list_fn = "%s/pf_60_mergelanes_bamfiles.txt" % output_dir
chromosomeMap_fn = "%s/chromosomeMap.tab" % output_dir
grc_properties_fn = "%s/grc/grc.properties" % output_dir
species_properties_fn = "%s/species/species.properties" % output_dir
submitArray_fn = "%s/grc/submitArray.sh" % output_dir
submitSpeciesArray_fn = "%s/species/submitArray.sh" % output_dir
runArrayJob_fn = "%s/grc/runArrayJob.sh" % output_dir
runSpeciesArrayJob_fn = "%s/species/runArrayJob.sh" % output_dir
mergeGrcResults_fn = "%s/grc/mergeGrcResults.sh" % output_dir
mergeSpeciesResults_fn = "%s/species/mergeSpeciesResults.sh" % output_dir

ref_fasta_fn = "/lustre/scratch116/malaria/pfalciparum/resources/Pfalciparum.genome.fasta"

bam_list_fn

get_ipython().system('cp /nfs/users/nfs_r/rp7/pf_60_mergelanes.txt {bam_fn}')

# Create list of bam files in format required
tbl_bam_file = (etl
    .fromtsv(bam_fn)
    .addfield('ChrMap', 'Pf3k')
    .rename('path', 'BamFile')
    .rename('sample', 'Sample')
    .cut(['Sample', 'BamFile', 'ChrMap'])
)
tbl_bam_file.totsv(bam_list_fn)
get_ipython().system('dos2unix -o {bam_list_fn}')

fo = open(grc_properties_fn, 'w')
print('''grc.loci=crt_core,crt_ex01,crt_ex02,crt_ex03,crt_ex04,crt_ex06,crt_ex09,crt_ex10,crt_ex11,dhfr_1,dhfr_2,dhfr_3,dhps_1,dhps_2,dhps_3,dhps_4,mdr1_1,mdr1_2,mdr1_3,arps10,mdr2,fd,exo

# CRT
grc.locus.crt_core.region=Pf3D7_07_v3:403500-403800
grc.locus.crt_core.targets=crt_72-76@403612-403626
grc.locus.crt_core.anchors=403593@TATTATTTATTTAAGTGTA,403627@ATTTTTGCTAAAAGAAC

grc.locus.crt_ex01.region=Pf3D7_07_v3:403150-404420
grc.locus.crt_ex01.targets=crt_24@403291-403293
grc.locus.crt_ex01.anchors=403273@GAGCGTTATA.[AG]GAATTA...AATTTA.TACAAGAA[GA]GAA

grc.locus.crt_ex02.region=Pf3D7_07_v3:403550-403820
grc.locus.crt_ex02.targets=crt_97@403687-403689
grc.locus.crt_ex02.anchors=403657@GGTAACTATAGTTTTGT.[AT]CATC[CT]GAAAC,403690@AACTTTATTTGTATGATTA[TA]GTTCTTTATT

grc.locus.crt_ex03.region=Pf3D7_07_v3:403850-404170
grc.locus.crt_ex03.targets=crt_144@404007-404009,crt_148@404019-404021
grc.locus.crt_ex03.anchors=404022@ACAAGAACTACTGGAAA[TC]AT[CT]CA[AG]TCATTT,403977@TC[CT]AT.TTA.AT[GT]CCTGTTCA.T[CA]ATT

grc.locus.crt_ex04.region=Pf3D7_07_v3:404200-404500
grc.locus.crt_ex04.targets=crt_194@404329-404331,crt_220@404407-404409
grc.locus.crt_ex04.anchors=404304@CGGAGCA[GC]TTATTATTGTTGTAACA...GCTC,404338@GTAGAAATGAAATTATC[TA]TTTGAAACAC,404359@GAAACACAAGAAGAAAATTCTATC[AG]TATTTAATC,404382@C[AG]TATTTAATCTTGTCTTA[AT]TTAGT...TTAATTG

grc.locus.crt_ex06.region=Pf3D7_07_v3:404700-405000
grc.locus.crt_ex06.targets=crt_271@404836-404838
grc.locus.crt_ex06.anchors=404796@TTGTCTTATATT.CCTGTATACACCCTTCCATT[TC]TTAAAA...C

grc.locus.crt_ex09.region=Pf3D7_07_v3:405200-405500
grc.locus.crt_ex09.targets=crt_326@405361-405363,crt_333@405382-405384
grc.locus.crt_ex09.anchors=405334@AAAACCTT[CT]G[CT]ATTGTTTTCCTTCTTT,405364@A.TTGTGATAATTTAATA...AGCTAT

grc.locus.crt_ex10.region=Pf3D7_07_v3:405400-405750
grc.locus.crt_ex10.targets=crt_342@405557-405559,crt_356@405599-405601
grc.locus.crt_ex10.anchors=405539@ATTATCGACAAATTTTCT...[AT]TGACATATAC,405573@TTGTTAGTTGTATACAAG[GT]TCCA[GA]CA,405602@GCAATT[GT]CTTATTACTTTAAATTCTTA[GA]CC

grc.locus.crt_ex11.region=Pf3D7_07_v3:405700-406000
grc.locus.crt_ex11.targets=crt_371@405837-405839
grc.locus.crt_ex11.anchors=405825@[GT]GTGATGTT.[TA]A...G.ACCAAGATTATTAG,405840@G.ACCAAGATTATTAGATTTCGTAACTTTG

# DHFR
grc.locus.dhfr_1.region=Pf3D7_04_v3:748100-748400
grc.locus.dhfr_1.targets=dhfr_51@748238-748240,dhfr_59@748262-748264
grc.locus.dhfr_1.anchors=748200@GAGGTCTAGGAAATAAAGGAGTATTACCATGGAA,748241@TCCCTAGATATGAAATATTTT...GCAG,748265@GCAGTTACAACATATGTGAATGAATC

grc.locus.dhfr_2.region=Pf3D7_04_v3:748250-748550
grc.locus.dhfr_2.targets=dhfr_108@748409-748411
grc.locus.dhfr_2.anchors=748382@CAAAATGTTGTAGTTATGGGAAGAACA,748412@TGGGAAAGCATTCCAAAAAAATTT

grc.locus.dhfr_3.region=Pf3D7_04_v3:748400-748720
grc.locus.dhfr_3.targets=dhfr_164@748577-748579
grc.locus.dhfr_3.anchors=748382@GGGAAATTAAATTACTATAAATG,748382@CTATAAATGTTTTATT...GGAGGTTC,748412@GGAGGTTCCGTTGTTTATCAAG


# DHPS
grc.locus.dhps_1.region=Pf3D7_08_v3:549550-549750
grc.locus.dhps_1.targets=dhps_436@549681-549683,dhps_437@549684-549686
grc.locus.dhps_1.anchors=549657@GTTATAGAT[AG]TAGGTGGAGAATCC,549669@GGTGGAGAATCC..TG.TCC,549687@CCTTTTGTTAT[AG]CCTAATCCAAAAATTAGTG

grc.locus.dhps_2.region=Pf3D7_08_v3:549850-550150
grc.locus.dhps_2.targets=dhps_540@549993-549995
grc.locus.dhps_2.anchors=549949@GTGTAGTTCTAATGCATAAAAGAGG,549970@GAGGAAATCCACATACAATGGAT,549985@CAATGGAT...CTAACAAATTA[TA]GATA,549996@CTAACAAATTA[TA]GATAATCTAGT

grc.locus.dhps_3.region=Pf3D7_08_v3:549950-550250
grc.locus.dhps_3.targets=dhps_581@550116-550118
grc.locus.dhps_3.anchors=550092@CTATTTGATATTGGATTAGGATTT,550119@AAGAAACATGATCAATCT[AT]TTAAACTC

grc.locus.dhps_4.region=Pf3D7_08_v3:550050-550350
grc.locus.dhps_4.targets=dhps_613@550212-550214
grc.locus.dhps_4.anchors=550167@GATGAGTATCCACTTTTTATTGG,550188@GGATATTCAAGAAAAAGATTTATT,550215@CATTGCATGAATGATCAAAATGTTG


# MDR1
grc.locus.mdr1_1.region=Pf3D7_05_v3:957970-958280
grc.locus.mdr1_1.targets=mdr1_86@958145-958147
grc.locus.mdr1_1.anchors=958120@GTTTG[GT]TGTAATATTAAA[GA]AACATG,958141@CATG...TTAGGTGATGATATTAATCCT

grc.locus.mdr1_2.region=Pf3D7_05_v3:958300-958600
grc.locus.mdr1_2.targets=mdr1_184@958439-958441
grc.locus.mdr1_2.anchors=958413@CATATGC[CA]AGTTCCTTTTTAGG,958446@GGTC[AG]TTAATAAAAAAT[GA]CACGTTTGAC

grc.locus.mdr1_3.region=Pf3D7_05_v3:961470-961770
grc.locus.mdr1_3.targets=mdr1_1246@961625-961627
grc.locus.mdr1_3.anchors=961595@GTTATAGAT[AG]TAGGTGGAGAATCC,961628@CTTAGAAA[CT][TA]TATTTTC[AT]ATAGTTAGTC

# ARPS10
grc.locus.arps10.region=Pf3D7_14_v3:2480900-2481200
grc.locus.arps10.targets=arps10_127@2481070-2481072
grc.locus.arps10.anchors=2481045@ATTTAC[CA]TTTTTGCGATCTCCCCAT...[GC],2481079@GACAGT[AC]G[AG]GA[GA]CAATTCGAAATAAAAC

# MDR2
grc.locus.mdr2.region=Pf3D7_14_v3:1956070-1956370
grc.locus.mdr2.targets=mdr2_484@-1956224-1956226
grc.locus.mdr2.anchors=1956203@ACATGTTATTAATCCT[TC]TAT...TGCC,1956227@TGCCGGAATAAT[AG]TACATTAAAACAGAAC

# Ferredoxin
grc.locus.fd.region=Pf3D7_13_v3:748250-748550
grc.locus.fd.targets=fd_193@-748393-748395
grc.locus.fd.anchors=748396@[GA]TGTAGTTCGTCTTCCTTGTG[CT]GTTTC

# Exo
grc.locus.exo.region=Pf3D7_13_v3:2504400-2504700
grc.locus.exo.targets=exo_415@2504559-2504561
grc.locus.exo.anchors=2504526@[GC]ATGATTTTA[AG][CA]AATATGGT[TC]ATAA[CT]GATAAAA,2504562@GAA[GT]TAAA[CT][AC]ATCATTGG[GA]AAAA[TC]AATATATAC
''', file=fo)
fo.close() 



fo = open(species_properties_fn, 'w')
print('''sampleClass.classes=Pf,Pv,Pm,Pow,Poc,Pk
sampleClass.loci=mito1,mito2,mito3,mito4,mito5,mito6 

sampleClass.locus.mito1.region=M76611:520-820 
sampleClass.locus.mito1.anchors=651@CCTTACGTACTCTAGCT....ACACAA
sampleClass.locus.mito1.targets=species1@668-671&678-683
sampleClass.locus.mito1.target.species1.alleles=Pf@ATGATTGTCT|ATGATTGTTT,Pv@TTTATATTAT,Pm@TTGTATTAAT,Pow@ATTTACATAA,Poc@ATTTATATAT,Pk@TTTTTATTAT

sampleClass.locus.mito2.region=M76611:600-900 
sampleClass.locus.mito2.anchors=741@GAATAGAA...GAACTCTATAAATAACCA
sampleClass.locus.mito2.targets=species2@728-733&740-740&749-751&770-773
sampleClass.locus.mito2.target.species2.alleles=Pf@GTTCATTTAAGATT|GTTCATTTAAGACT,Pv|Pk@TATTCATAAATACA,Pm@GTTCAATTAGTACT,Pow|Poc@GTTACAATAATATT

sampleClass.locus.mito3.region=M76611:720-1020 
sampleClass.locus.mito3.anchors=842@(?:GAAAGAATTTATAA|ATATA[AG]TGAATATG)ACCAT
sampleClass.locus.mito3.targets=species3@861-869&878-881&884-887
sampleClass.locus.mito3.target.species3.alleles=Pf@TCGGTAGAATATTTATT,Pv@TCACTATTACATTAACT,Pm@TCACTATTTAATATATC,Pow@CCCTTATTTAACTAACC|TCCTTATTTAACTAACC,Poc@TCGTTATTAAACTAACC,Pk@TCACAATTAAACTTATT

sampleClass.locus.mito4.region=M76611:820-1120 
sampleClass.locus.mito4.anchors=948@CCTGTAACACAATAAAATAATGT
sampleClass.locus.mito4.targets=species4@971-982
sampleClass.locus.mito4.target.species4.alleles=Pf@AGTATATACAGT,Pv|Pow|Poc@ACCAGATATAGC,Pm@TCCTGAAACTCC,Pk@ACCTGATATAGC

sampleClass.locus.mito5.region=M76611:900-1200 
sampleClass.locus.mito5.anchors=1029@GATGCAAAACATTCTCC
sampleClass.locus.mito5.targets=species5@1025-1028&1046-1049
sampleClass.locus.mito5.target.species5.alleles=Pf@TAGATAAT,Pv|Pk@AAGTAAGT,Pm@TAATAAGT,Pow@TAATAAGA,Poc@TAATAAGG

sampleClass.locus.mito6.region=M76611:950-1250
sampleClass.locus.mito6.anchors=1077@ATTTC[AT]AAACTCAT[TA]CCTTTTTCTA
sampleClass.locus.mito6.targets=species6@1062-1066&1073-1073&1076-1076&1082-1082&1091-1091&1102-1108
sampleClass.locus.mito6.target.species6.alleles=Pf@CAAATAGATTAAATAC,Pv|Pk@AATACAATTTTAGAAA|AATATAATTTTAGAAA,Pm@AATATTTAAAAAGAAA,Pow|Poc@AATATTTTTTGAGAAA|AATATTTTTTAAGAAA
''', file=fo)
fo.close() 

fo = open(chromosomeMap_fn, 'w')
print('''default	Pf3k
Pf3D7_01_v3	Pf3D7_01_v3
Pf3D7_02_v3	Pf3D7_02_v3
Pf3D7_03_v3	Pf3D7_03_v3
Pf3D7_04_v3	Pf3D7_04_v3
Pf3D7_05_v3	Pf3D7_05_v3
Pf3D7_06_v3	Pf3D7_06_v3
Pf3D7_07_v3	Pf3D7_07_v3
Pf3D7_08_v3	Pf3D7_08_v3
Pf3D7_09_v3	Pf3D7_09_v3
Pf3D7_10_v3	Pf3D7_10_v3
Pf3D7_11_v3	Pf3D7_11_v3
Pf3D7_12_v3	Pf3D7_12_v3
Pf3D7_13_v3	Pf3D7_13_v3
Pf3D7_14_v3	Pf3D7_14_v3
M76611	Pf_M76611
PFC10_API_IRAB	Pf3D7_API_v3
''', file=fo)
fo.close()

fo = open(runArrayJob_fn, 'w')
print('''BAMLIST_FILE=$1
CONFIG_FILE=$2
REF_FASTA_FILE=$3
CHR_MAP_FILE=$4
OUT_DIR=$5
 
JOB=$LSB_JOBINDEX
#JOB=3
 
IN=`sed "$JOB q;d" $BAMLIST_FILE`
read -a LINE <<< "$IN"
SAMPLE_NAME=${LINE[0]}
BAM_FILE=${LINE[1]}
CHR_MAP_NAME=${LINE[2]}
 
JAVA_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101
JRE_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101

GRCC=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/AnalysisCommon
GRCA=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/SequencingReadsAnalysis
CLASSPATH=$GRCA/bin:$GRCC/bin:\
$GRCC/lib/commons-logging-1.1.1.jar:\
$GRCA/lib/apache-ant-1.8.2-bzip2.jar:\
$GRCA/lib/commons-compress-1.4.1.jar:\
$GRCA/lib/commons-jexl-2.1.1.jar:\
$GRCA/lib/htsjdk-2.1.0.jar:\
$GRCA/lib/ngs-java-1.2.2.jar:\
$GRCA/lib/snappy-java-1.0.3-rc3.jar:\
$GRCA/lib/xz-1.5.jar

$JAVA_HOME/bin/java -cp $CLASSPATH -Xms512m -Xmx2000m 'org.cggh.bam.grc.GrcAnalysis$SingleSample' $CONFIG_FILE $SAMPLE_NAME $BAM_FILE $CHR_MAP_NAME $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''', file=fo)
fo.close()

fo = open(runSpeciesArrayJob_fn, 'w')
print('''BAMLIST_FILE=$1
CONFIG_FILE=$2
REF_FASTA_FILE=$3
CHR_MAP_FILE=$4
OUT_DIR=$5
 
JOB=$LSB_JOBINDEX
#JOB=3
 
IN=`sed "$JOB q;d" $BAMLIST_FILE`
read -a LINE <<< "$IN"
SAMPLE_NAME=${LINE[0]}
BAM_FILE=${LINE[1]}
CHR_MAP_NAME=${LINE[2]}
 
JAVA_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101
JRE_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101

GRCC=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/AnalysisCommon
GRCA=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/SequencingReadsAnalysis
CLASSPATH=$GRCA/bin:$GRCC/bin:\
$GRCC/lib/commons-logging-1.1.1.jar:\
$GRCA/lib/apache-ant-1.8.2-bzip2.jar:\
$GRCA/lib/commons-compress-1.4.1.jar:\
$GRCA/lib/commons-jexl-2.1.1.jar:\
$GRCA/lib/htsjdk-2.1.0.jar:\
$GRCA/lib/ngs-java-1.2.2.jar:\
$GRCA/lib/snappy-java-1.0.3-rc3.jar:\
$GRCA/lib/xz-1.5.jar

echo
echo $SAMPLE_NAME
echo $BAM_FILE
echo $CHR_MAP_NAME
echo
echo $JAVA_HOME/bin/java -cp $CLASSPATH -Xms512m -Xmx2000m 'org.cggh.bam.sampleClass.SampleClassAnalysis$SingleSample' $CONFIG_FILE $SAMPLE_NAME $BAM_FILE $CHR_MAP_NAME $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
echo

$JAVA_HOME/bin/java -cp $CLASSPATH -Xms512m -Xmx2000m 'org.cggh.bam.sampleClass.SampleClassAnalysis$SingleSample' $CONFIG_FILE $SAMPLE_NAME $BAM_FILE $CHR_MAP_NAME $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''', file=fo)
fo.close()

fo = open(submitArray_fn, 'w')
print('''BAMLIST_FILE=%s
CONFIG_FILE=%s
REF_FASTA_FILE=%s
CHR_MAP_FILE=%s
OUT_DIR=%s/grc
LOG_DIR=%s/log
 
NUM_BAMLIST_LINES=`wc -l < $BAMLIST_FILE`
QUEUE=normal
# NUM_BAMLIST_LINES=2
# QUEUE=small

bsub -q $QUEUE -G malaria-dk -J "genotype[2-$NUM_BAMLIST_LINES]%%25" -R"select[mem>2000] rusage[mem=2000] span[hosts=1]" -M 2000 -o $LOG_DIR/output_%%J-%%I.log %s $BAMLIST_FILE $CONFIG_FILE $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''' % (
        bam_list_fn,
        grc_properties_fn,
        ref_fasta_fn,
        chromosomeMap_fn,
        scratch_dir,
        scratch_dir,
        "bash %s" % runArrayJob_fn,
        ),
     file=fo)
fo.close()

fo = open(submitSpeciesArray_fn, 'w')
print('''BAMLIST_FILE=%s
CONFIG_FILE=%s
REF_FASTA_FILE=%s
CHR_MAP_FILE=%s
OUT_DIR=%s/species
LOG_DIR=%s/log
 
NUM_BAMLIST_LINES=`wc -l < $BAMLIST_FILE`
QUEUE=small
# NUM_BAMLIST_LINES=2
# QUEUE=small

bsub -q $QUEUE -G malaria-dk -J "genotype[2-$NUM_BAMLIST_LINES]%%25" -R"select[mem>2000] rusage[mem=2000] span[hosts=1]" -M 2000 -o $LOG_DIR/output_%%J-%%I.log %s $BAMLIST_FILE $CONFIG_FILE $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''' % (
        bam_list_fn,
        species_properties_fn,
        ref_fasta_fn,
        chromosomeMap_fn,
        scratch_dir,
        scratch_dir,
        "bash %s" % runSpeciesArrayJob_fn,
        ),
     file=fo)
fo.close()

fo = open(mergeGrcResults_fn, 'w')
print('''BAMLIST_FILE=%s
CONFIG_FILE=%s
REF_FASTA_FILE=%s
CHR_MAP_FILE=%s
OUT_DIR=%s/grc
 
JAVA_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101
JRE_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101

GRCC=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/AnalysisCommon
GRCA=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/SequencingReadsAnalysis
CLASSPATH=$GRCA/bin:$GRCC/bin:\
$GRCC/lib/commons-logging-1.1.1.jar:\
$GRCA/lib/apache-ant-1.8.2-bzip2.jar:\
$GRCA/lib/commons-compress-1.4.1.jar:\
$GRCA/lib/commons-jexl-2.1.1.jar:\
$GRCA/lib/htsjdk-2.1.0.jar:\
$GRCA/lib/ngs-java-1.2.2.jar:\
$GRCA/lib/snappy-java-1.0.3-rc3.jar:\
$GRCA/lib/xz-1.5.jar

$JAVA_HOME/bin/java -cp $CLASSPATH -Xms512m -Xmx2000m 'org.cggh.bam.grc.GrcAnalysis$MergeResults' $CONFIG_FILE $BAMLIST_FILE $REF_FASTA_FILE $CHR_MAP_FILE $OUT_DIR
''' % (
        bam_list_fn,
        grc_properties_fn,
        ref_fasta_fn,
        chromosomeMap_fn,
        scratch_dir,
        ),
     file=fo)
fo.close()

fo = open(mergeSpeciesResults_fn, 'w')
print('''BAMLIST_FILE=%s
CONFIG_FILE=%s
REF_FASTA_FILE=%s
OUT_DIR=%s/species
 
JAVA_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101
JRE_HOME=/nfs/team112_internal/rp7/opt/java/jdk1.8.0_101

GRCC=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/AnalysisCommon
GRCA=/nfs/team112_internal/rp7/src/github/malariagen/GeneticReportCard/SequencingReadsAnalysis
CLASSPATH=$GRCA/bin:$GRCC/bin:\
$GRCC/lib/commons-logging-1.1.1.jar:\
$GRCA/lib/apache-ant-1.8.2-bzip2.jar:\
$GRCA/lib/commons-compress-1.4.1.jar:\
$GRCA/lib/commons-jexl-2.1.1.jar:\
$GRCA/lib/htsjdk-2.1.0.jar:\
$GRCA/lib/ngs-java-1.2.2.jar:\
$GRCA/lib/snappy-java-1.0.3-rc3.jar:\
$GRCA/lib/xz-1.5.jar

$JAVA_HOME/bin/java -cp $CLASSPATH -Xms512m -Xmx2000m 'org.cggh.bam.species.SpeciesAnalysis$MergeResults' $CONFIG_FILE $BAMLIST_FILE $REF_FASTA_FILE $OUT_DIR
''' % (
        bam_list_fn,
        species_properties_fn,
        ref_fasta_fn,
        scratch_dir,
        ),
     file=fo)
fo.close()

get_ipython().system('bash {submitArray_fn}')

get_ipython().system('bash {mergeGrcResults_fn}')

get_ipython().system('bash {submitSpeciesArray_fn}')

submitSpeciesArray_fn

get_ipython().system('bash {mergeSpeciesResults_fn}')

