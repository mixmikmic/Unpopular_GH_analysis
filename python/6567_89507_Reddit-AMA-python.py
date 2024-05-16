from pyspark import SparkContext, SparkConf

sqlContext = SQLContext(sc)

df = sqlContext.read.format("com.cloudant.spark").option("cloudant.host","XXXX").option("cloudant.username", "XXXX").option("cloudant.password","XXXX").option("schemaSampleSize", "-1").load("reddit_ibmama_top_comments_only")

df.printSchema()

df.show()

df.registerTempTable("reddit");

sentimentDistribution=[0] * 13

for i, sentiment in enumerate(df.columns[-23:13]):
    sentimentDistribution[i]=sqlContext.sql("SELECT count(*) as sentCount FROM reddit where cast(" + sentiment + " as String) > 70.0")        .collect()[0].sentCount

print sentimentDistribution

get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
 
ind=np.arange(13)
width = 0.35
bar = plt.bar(ind, sentimentDistribution, width, color='g', label = "distributions")
 
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*3.5, plSize[1]*2) )
plt.ylabel('Reddit comment count')
plt.xlabel('Emotional Tone')
plt.title('Histogram of comments by sentiments > 70% in IBM Reddit AMA')
plt.xticks(ind+width, df.columns[-23:13])
plt.legend()
 
plt.show()

comments=[]

for i, sentiment in enumerate(df.columns[-23:13]):
    commentset = df.filter("cast(" + sentiment + " as String) > 70.0")
    comments.append(commentset.map(lambda p: p.author + "\n\n" + p.text).collect())
    print "\n--------------------------------------------------------------------------------------------"
    print sentiment
    print "--------------------------------------------------------------------------------------------\n"
    for comment in comments[i]:
        print "[-]  " + comment +"\n"
    



