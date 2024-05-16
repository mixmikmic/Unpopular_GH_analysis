import psycopg2
import numpy as np

conn = psycopg2.connect(dbname='forest_fires')
c = conn.cursor()

# Grab the fire names for 2013 that have the highest number of obs per fire_name and date.
c.execute('''SELECT COUNT(fire_name) as total, fire_name, date_
            FROM daily_fire_shapefiles_2013 
            GROUP BY fire_name, date_
            ORDER BY total DESC
            LIMIT 20; ''')
c.fetchall()

# Now let's look at a a couple of these and see whats different. SELECT * 
# won't work below because there is a field that returns all blanks and 
# causes an error. 
columns = ['acres', 'agency', 'time_', 'comments', 'year_', 'active', 
          'unit_id', 'fire_num', 'fire', 'load_date', 'inciweb_id', 
          'st_area_sh', 'st_length_', 'st_area__1', 'st_length1', 
          'st_area__2', 'st_lengt_1']
for column in columns: 
    c.execute('''SELECT ''' + column + '''
                FROM daily_fire_shapefiles_2013 
                WHERE fire_name = 'Douglas Complex' and date_ = '2013-8-4'; ''')
    print column + ':', np.unique(c.fetchall())

# Grab the fire names for 2014 that have the highest number of obs per fire_name and date.
c.execute('''SELECT COUNT(fire_name) as total, fire_name, date_
            FROM daily_fire_shapefiles_2014 
            GROUP BY fire_name, date_
            ORDER BY total DESC
            LIMIT 20; ''')
c.fetchall()

# Now let's look at a a couple of these and see whats different. SELECT * 
# won't work below because there is a field that returns all blanks and 
# causes an error. The columns also aren't the same in 2014 as they are in 2013. 
columns = ['acres', 'agency', 'time_', 'comments', 'year_', 'active', 
          'unit_id', 'fire_num', 'fire', 'load_date', 'inciweb_id', 
          'st_area_sh', 'st_length_']
for column in columns: 
    c.execute('''SELECT ''' + column + '''
                FROM daily_fire_shapefiles_2014 
                WHERE fire_name = 'July Complex' and date_ = '2014-8-27'; ''')
    print column + ':', np.unique(c.fetchall())



