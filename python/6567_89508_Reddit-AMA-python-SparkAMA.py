from pyspark import SparkContext, SparkConf

sqlContext = SQLContext(sc)

df = sqlContext.read.format("com.cloudant.spark").option("cloudant.host","c51d6693-95f1-4e8c-89c6-332bb8cf6ec5-bluemix.cloudant.com").option("cloudant.username", "c51d6693-95f1-4e8c-89c6-332bb8cf6ec5-bluemix").option("cloudant.password","df29c8ff5116a6b8c8a01e1e296059580133859b79d95f674736980d420e9431").option("schemaSampleSize", "-1").load("reddit_spark-ama_top_comments_only")

df.printSchema()

df.show()

df.registerTempTable("reddit");

sentimentDistribution=[0] * 13
#for i, sentiment in enumerate(df.columns[-23:13]): print sentiment

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
htmlbody = ""

for i, sentiment in enumerate(df.columns[-23:13]):
    commentset = df.filter("cast(" + sentiment + " as String) > 70.0")
    comments.append(commentset.map(lambda p: p.author + "\n\n<p>" + p.text).collect())
    htmlbody = htmlbody + "<div class=\"container\">\n<div class=\"panel-group\">\n<div class=\"panel panel-default\">\n"
    htmlbody = htmlbody + "<div class=\"panel-heading\">" + "\n" + "<h4 class=\"panel-title\">" + "\n" + "<a data-toggle=\"collapse\" href=\"#" + sentiment + "\">" + sentiment + "</a>\n</h4>\n</div>"
    htmlbody = htmlbody + "<div id=" + sentiment + " class=\"panel-collapse collapse in\">" + "\n"
    for comment in comments[i]:
         htmlbody = htmlbody + "<div class=\"panel-body\">" + "[-]  " + comment + "\n</div>"
    htmlbody = htmlbody + "\n</div>\n</div>\n</div>\n</div>\n"    
#print htmlbody

get_ipython().run_cell_magic(u'HTML', u'',u'<head>'
u'<meta name="viewport" content="width=device-width, initial-scale=1">'
u'<link rel="stylesheet" href="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css">'
u'<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.0/jquery.min.js"></script>'
  u'<script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js"></script>'
u'</head>'
u'<body>'
u''
    + htmlbody +
u''
u'</body>')



