dir()

from pyspark.sql import HiveContext
from pyspark.sql import Row

hiveCtx = HiveContext(sc)
lines = hiveCtx.read.json("boston_university.json")

