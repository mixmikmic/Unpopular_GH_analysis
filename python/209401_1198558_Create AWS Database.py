import psycopg2
import sys
import numpy as np
import pandas as pd
import pandas.io.sql as pdsql
import sqlalchemy as sq

def connect_to_db_sqlalchemy():
    conn_string = "postgresql+psycopg2://"+SENS.dsn_uid+":"+SENS.dsn_pwd+"@"+SENS.dsn_hostname+":"+SENS.dsn_port+"/"+SENS.dsn_database
    engine = sq.create_engine(conn_string)
    return engine

def connect_to_db():
    import SENSITIVE as SENS
    try:
        conn_string = "host="+SENS.dsn_hostname+" port="+SENS.dsn_port+" dbname="+SENS.dsn_database+" user="+SENS.dsn_uid+" password="+SENS.dsn_pwd
        print("Connecting to database\n  ->%s" % (conn_string.replace(SENS.dsn_pwd, '#'*len(SENS.dsn_pwd))))
        conn=psycopg2.connect(conn_string)
        print("Connected!\n")
        return conn
    except:
        print("Unable to connect to the database.")
        return

conn = connect_to_db()
# conn.close()

# Print available databases:
cursor = conn.cursor()
cursor.execute("""SELECT datname from pg_database""")
rows = cursor.fetchall()
print("\nShow me the databases:\n")
for row in rows:
    print("   ", row[0])



conn = connect_to_db()
cursor = conn.cursor()
table_name = 'title_basics'
query = """CREATE TABLE %s(tconst VARCHAR(20) PRIMARY KEY, 
                           titleType VARCHAR(150), 
                           primaryTitle VARCHAR(500), 
                           originalTitle VARCHAR(500), 
                           isAdult BOOLEAN, 
                           startYear INTEGER, 
                           endYear INTEGER, 
                           runtimeMinutes INTEGER, 
                           genre VARCHAR(150))""" % table_name

cursor.execute("DROP TABLE IF EXISTS %s" % table_name)
cursor.execute(query)
conn.commit()
conn.close()

title_basics_df = pd.read_csv('Data/title.basics.tsv', sep='\t')
def clean_year(y):
    import numpy as np
    try:
        return int(y)
    except:
        return -9999

def clean_genre(y):
    y = str(y)
    if y == '\\N':
        return ''
    return y.split(',')[0].strip()
import datetime
import numpy as np
print(len(title_basics_df))
# title_basics_df.drop('endYear', axis=1, inplace=True)
title_basics_df['endYear'] = title_basics_df['endYear'].apply(clean_year)
title_basics_df['startYear'] = title_basics_df['startYear'].apply(clean_year)
title_basics_df['runtimeMinutes'] = title_basics_df['runtimeMinutes'].apply(clean_year)
title_basics_df['genres'] = title_basics_df['genres'].apply(clean_genre)
title_basics_df['isAdult'] = title_basics_df['isAdult'].apply(bool)
title_basics_df.head()

engine = connect_to_db_sqlalchemy()

title_basics_df.iloc[:100].to_sql(table_name, engine)





def generate_query(table_name, df):
    query = "INSERT INTO " + table_name + " VALUES "
    for i, row in df.iterrows():
        query += str(tuple(row.values))
        query += ', '
    return query[:-2]

s = generate_query(table_name, title_basics_df.iloc[:100])
print(s)

conn = connect_to_db()
cursor = conn.cursor()
cursor.execute(s)
cursor.commit()
conn.close()



table_name = 'title_crew'
query = """CREATE TABLE %s(tconst VARCHAR(20) PRIMARY KEY, 
                           directors VARCHAR(500), 
                           writers VARCHAR(500))""" % table_name

cursor.execute("DROP TABLE IF EXISTS %s" % table_name)
cursor.execute(query)
conn.commit()

table_name = 'title_episode'
query = """CREATE TABLE %s(tconst VARCHAR(20) PRIMARY KEY, 
                           parentTconst VARCHAR(20), 
                           seasonNumber INTEGER,
                           episodeNumber INTEGER)""" % table_name

cursor.execute("DROP TABLE IF EXISTS %s" % table_name)
cursor.execute(query)
conn.commit()

table_name = 'title_principals'
query = """CREATE TABLE %s(tconst VARCHAR(20) PRIMARY KEY, 
                           principalCast VARCHAR(500))""" % table_name

cursor.execute("DROP TABLE IF EXISTS %s" % table_name)
cursor.execute(query)
conn.commit()

table_name = 'title_ratings'
query = """CREATE TABLE %s(tconst VARCHAR(20) PRIMARY KEY, 
                           averageRating REAL,
                           numVotes INTEGER)""" % table_name

cursor.execute("DROP TABLE IF EXISTS %s" % table_name)
cursor.execute(query)
conn.commit()

table_name = 'name_basics'
query = """CREATE TABLE %s(nconst VARCHAR(20) PRIMARY KEY, 
                           prmaryName VARCHAR(250),
                           birthYear INTEGER,
                           deathYear INTEGER,
                           primaryProfession VARCHAR(500),
                           knownForTitles VARCHAR(500))""" % table_name

cursor.execute("DROP TABLE IF EXISTS %s" % table_name)
cursor.execute(query)
conn.commit()







