import findspark
findspark.init()

import pyspark
sc=pyspark.SparkContext()

sqlCtx = pyspark.SQLContext(sc)

#Reads the json file into a dataframe
df = sqlCtx.read.json("census_2010.json")
print(type(df))

#Prints the schema of the columns
df.printSchema()

#prints the first 20 rows
df.show()

first_five = df.head(5)[0:5]
for element in first_five:
    print(element)

first_one = df.head(5)[0]
print(first_one)

#Selecting columns from spark dataframes and display
df.select('age', 'males', 'females', 'year').show()

#Using boolean filtering and select rows where age > 5
five_plus = df[df['age'] > 5]
five_plus.show()

#Shows all columns where females < males
df[df['females'] < df['males']].show()

import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

pandas_df = df.toPandas()
pandas_df['total'].hist()

pandas_df.head()

#Register a temp table

df.registerTempTable('census2010')
tables = sqlCtx.tableNames()
print(tables)

q1 = "SELECT age FROM census2010"

sqlCtx.sql(q1).show()

q2 = "SELECT males, females FROM census2010 WHERE age > 5 AND age < 15"

sqlCtx.sql(q2).show()

#Using describe to show basic statistics
q3 = "SELECT males, females FROM census2010"

sqlCtx.sql(q3).describe().show()

#Load files into the sqlCtx object
df = sqlCtx.read.json("census_2010.json")
df2 = sqlCtx.read.json("census_1980.json")
df3 = sqlCtx.read.json("census_1990.json")
df4 = sqlCtx.read.json("census_2000.json")

df.registerTempTable('census2010')
df2.registerTempTable('census1980')
df3.registerTempTable('census1990')
df4.registerTempTable('census2000')

#Shows the table names
tables = sqlCtx.tableNames()
print(tables)

#Using joins with sqlCtx

q4 = 'SELECT c1.total, c2.total FROM census2010 c1 INNER JOIN census2000 c2 ON c1.age = c2.age'


sqlCtx.sql(q4).show()

#Using SQL aggregate functions with multiple files

q5 = '''
SELECT 
    SUM(c1.total) 2010_total, 
    SUM(c2.total) 2000_total,
    SUM(c3.total) 1990_total
FROM census2010 c1 
INNER JOIN census2000 c2 ON c1.age = c2.age
INNER JOIN census1990 c3 ON c1.age = c3.age


'''


sqlCtx.sql(q5).show()

