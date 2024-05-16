#from pyspark.sql import SQLContext
#sqlCtx = SQLContext(sc)

from pyspark.sql import HiveContext
from pyspark.sql import Row

hiveCtx = HiveContext(sc)
lines = hiveCtx.read.json("boston_university.json")
lines.registerTempTable("events")
lines.printSchema()

topTweets = hiveCtx.sql("""SELECT date_time, location, credit_url FROM events LIMIT 10""")

type(lines)

type(topTweets)

topTweets.show()

#from pyspark.sql.types import IntegerType
#hiveCtx.registerFunction("numChars", lambda line: len(line), IntegerType())

modified_url = topTweets.rdd.map(lambda x: (x[0], len(x[2])))
modified_url.collect()

topTweets.select(["date_time"]).show()

topTweets.filter(topTweets["location"] != 'null').select("location").show()

hiveCtx.sql("""SELECT location FROM events WHERE location != "null" LIMIT 10""").show()

topTweets.printSchema()

recs = [["Frank", 45, 180.4], ["Rocco", 23, 212.0], ["Claude", 38, 112.9]]
my_rdd = sc.parallelize(recs)

df = hiveCtx.createDataFrame(my_rdd)
df.show()

df.select("_2").show()

recsRow = [Row(name="Frank", age=45, weight=180.4),
           Row(name="Rocco", age=23, weight=212.0),
           Row(name="Claude", age=38, weight=112.9)]

df2 = hiveCtx.createDataFrame(recsRow)
df2.show()

df2.select("name", df2.age + 1).show()

df2.registerTempTable("Guys")
hiveCtx.sql("""SELECT name, age + 1 FROM Guys""").show()

