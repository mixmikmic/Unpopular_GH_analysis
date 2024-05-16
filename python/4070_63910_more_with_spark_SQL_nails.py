from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.master("local").appName("Nail Play").getOrCreate()

# The data is permits managed by the Boston Public Health Commission
nails = spark.read.csv('Nail_Salon_Permits.csv', header=True, inferSchema=True)
nails.printSchema()

nails.first()

nails.show(3)

nails.describe(['Number Tables', 'Salon_BusinessID']).show()

nails.select('*').filter(~F.isnull('Number Tables')).count()

nails.select('*').filter(F.isnull('Number Tables')).count()

nails.select('*').filter(F.isnull('Salon Neighborhood')).count()

nails.select(['Salon_BusinessID', 'SalonName', 'Salon Neighborhood', 'Number Tables', 'Salon_First_Visit']).orderBy('Salon_BusinessID').show(truncate=False)

nails.select(['Salon Neighborhood']).groupby('Salon Neighborhood').agg(F.count('*').alias('count')).show()

nails.select(['*']).groupby('Salon Neighborhood').agg(F.round(F.avg('Number Tables'), 1).alias('Avg Num Tables')).orderBy('Avg Num Tables', ascending=False).show(5)

nails.select('SalonName').filter(nails['Services Hair'] == 1).filter(nails['Services Wax'] == 1).distinct().orderBy('SalonName').show(5)



