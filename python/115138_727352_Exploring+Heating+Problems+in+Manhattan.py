import os
import matplotlib.pyplot as plt
import pandas as pd
os.environ['BRUNEL_CONFIG'] = "locjavascript=/data/jupyter2/static-file-content-delivery-network/nbextensions/brunel_ext"
import brunel
# Brunel will be used to visualize data later on

get_ipython().system('pip install brunel')

from pyspark.sql import SparkSession

# @hidden_cell
# This function is used to setup the access of Spark to your Object Storage. The definition contains your credentials.
# You might want to remove those credentials before you share your notebook.
def set_hadoop_config_with_credentials_65b20aee057f4804b65dcbe3451d97f5(name):
    """This function sets the Hadoop configuration so it is possible to
    access data from Bluemix Object Storage using Spark"""

    prefix = 'fs.swift.service.' + name
    hconf = sc._jsc.hadoopConfiguration()
    hconf.set(prefix + '.auth.url', 'https://identity.open.softlayer.com'+'/v3/auth/tokens')
    hconf.set(prefix + '.auth.endpoint.prefix', 'endpoints')
    hconf.set(prefix + '.tenant', '37b0da6beda44a5b961e39dbbae86eba')
    hconf.set(prefix + '.username', '098abbdf203242ecac90136bcb782360')
    hconf.set(prefix + '.password', 'F5t~6y.OQ_ee]c9m')
    hconf.setInt(prefix + '.http.port', 8080)
    hconf.set(prefix + '.region', 'dallas')
    hconf.setBoolean(prefix + '.public', False)

#hconf.set(prefix + '.tenant', '37b0da6beda44a5b961e39dbbae86eba')
#hconf.set(prefix + '.username', '098abbdf203242ecac90136bcb782360')
#hconf.set(prefix + '.password', 'F5t~6y.OQ_ee]c9m')
    
    
# you can choose any name
name = 'keystone'
set_hadoop_config_with_credentials_65b20aee057f4804b65dcbe3451d97f5(name)

#spark = SparkSession.builder.getOrCreate()

#df_data_1 = spark.read\
#  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')\
#  .option('header', 'true')\
#  .load('swift://IAETutorialsforWDPZBeta.' + name + '/IAE_examples_data_311NYC.csv')
#df_data_1.show(2)

spark = SparkSession.builder.getOrCreate()

nyc311DF = spark.read    .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')    .option('header', 'true')    .load('swift://IAETutorialsforWDPZBeta.' + name + '/IAE_examples_data_311NYC.csv')
nyc311DF.show(1)

nyc311DF.count()

nyc311DF.printSchema()

nyc311DF.createOrReplaceTempView("nyc311ct")

spark.sql("select distinct Borough from nyc311ct").show()

nyc311Agr_df = spark.sql("select `Complaint Type` as Complaint_Type, count(`Unique Key`) as Complaint_Count "
                            "from nyc311ct where Borough = 'MANHATTAN' "
                            "group by `Complaint Type` order by Complaint_Count desc").cache()

nyc311Agr_df.show(4)

#custom_frame = nyc311Agr_df.groupBy('Complaint_Type').count().sort('count').toPandas()
custom_frame = nyc311Agr_df.toPandas()
custom_frame.head(4)

get_ipython().magic("brunel data('custom_frame') bubble size(Complaint_Count) color(Complaint_Type) label(Complaint_Type) legends(none) tooltip(Complaint_Type)")
#%brunel data('custom_frame') x(Complaint_Type) y(count) chord size(count) :: width=500, height=400
#%brunel data ('custom_frame') bar x (Complaint_Type) y (count)

spark.sql("select `Incident Zip` as Zip, count(*) as ZipHeatingCnt " 
          "from nyc311ct " 
          "where `Complaint Type` = 'HEAT/HOT WATER' and `Incident Zip` <> '' group by `Incident Zip`").show()

spark.sql("select `Incident Zip` as Zip, count(*) as ZipHeatingCnt "  
          "from nyc311ct " 
          "where `Complaint Type` = 'HEAT/HOT WATER' and `Incident Zip` <> '' group by `Incident Zip`").createOrReplaceTempView("zipHeatingCnt")

spark.sql("select split(`Created Date`, ' ')[0] as Incident_Date, `Incident Zip` as Incident_Zip, "
          "count(`Unique Key`) as HeatingComplaintCount "
          "from nyc311ct where `Complaint Type` = 'HEAT/HOT WATER' and `Incident Zip` <> '' "
          "and split(split(`Created Date`, ' ')[0], '/')[2] = '16' "
          "group by split(`Created Date`, ' ')[0], `Incident Zip` order by HeatingComplaintCount desc limit 50").show()



