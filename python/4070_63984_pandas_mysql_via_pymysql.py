import os
import pymysql
import pandas as pd
from pandas.io import sql

host = 'localhost'
user = 'jhalverson'
passwd = os.environ['MYSQL_PASSWORD']
db = 'test'
con = pymysql.connect(host=host, user=user, passwd=passwd, db=db)

df = sql.read_sql('''select * from suppliers;''', con=con)
df

df.Supplier

df.info()

con.close()

