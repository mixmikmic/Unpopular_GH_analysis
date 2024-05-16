from pyspark import SparkContext, SparkConf

sqlContext = SQLContext(sc)

df = sqlContext.read.format("com.cloudant.spark").option("cloudant.host","XXXXX-bluemix.cloudant.com").option("cloudant.username", "XXXXX-bluemix").option("cloudant.password","XXXXX").load("reddit_redditama_all_posts")

df.printSchema()

df.show()

df.registerTempTable("reddit");

sentimentDistribution=[0] * 9
#for i, sentiment in enumerate(df.columns[-18:9]): print sentiment

for i, sentiment in enumerate(df.columns[-18:9]):
    sentimentDistribution[i]=sqlContext.sql("SELECT count(*) as sentCount FROM reddit where cast(" + sentiment + " as String) > 70.0")        .collect()[0].sentCount

print sentimentDistribution

get_ipython().magic('matplotlib inline')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
 
ind=np.arange(9)
width = 0.35
bar = plt.bar(ind, sentimentDistribution, width, color='g', label = "distributions")
 
params = plt.gcf()
plSize = params.get_size_inches()
params.set_size_inches( (plSize[0]*2.5, plSize[1]*2) )
plt.ylabel('Reddit comment count')
plt.xlabel('Emotion Tone')
plt.title('Histogram of comments by sentiments > 70% in IBM Reddit AMA')
plt.xticks(ind+width, df.columns[-18:9])
plt.legend()
 
plt.show()

comments=[]
#comments.append([])
for i, sentiment in enumerate(df.columns[-18:9]):
    commentset = df.filter("cast(" + sentiment + " as String) > 70.0")
    comments.append(commentset.map(lambda p: p.author + "\n\n" + p.text).collect())
    print "\n--------------------------------------------------------------------------------------------"
    print sentiment
    print "--------------------------------------------------------------------------------------------\n"
    for comment in comments[i]:
        print "[-]  " + comment +"\n"
    



