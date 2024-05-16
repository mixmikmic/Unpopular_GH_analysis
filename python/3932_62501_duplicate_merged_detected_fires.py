import psycopg2
import numpy as np

# First we need to merge on the perimeter boundary files, and then figure out which detected
# fire centroids have merged on in mutiple places. In the detected fire data, a unique 
# index is (lat, long, date, gmt, src). We'll start with 2013. 
conn = psycopg2.connect('dbname=forest_fires')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE merged_2013 AS
             (SELECT points.*, polys.fire_name, polys.fire, polys.agency, polys.unit_id
             FROM detected_fires_2013 as points
                    LEFT JOIN daily_fire_shapefiles_2013 as polys
             ON points.date = polys.date_ 
                AND ST_WITHIN(points.wkb_geometry, polys.wkb_geometry));
            ''')
conn.commit()

# Just to display what I'm talking about: 
cursor.execute('''SELECT COUNT(*)
                FROM detected_fires_2013;''')
print 'Detected_fires_2013 obs: ', cursor.fetchall()[0][0]

cursor.execute('''SELECT COUNT(*)
                FROM merged_2013;''')
print 'Merged_2013 obs: ', cursor.fetchall()[0][0]

# Let's check if any obs. now have more than one row per (lat, long, date, gmt, and src), which
# prior to this merge was unique. 
cursor.execute('''SELECT COUNT(*) as totals
                FROM merged_2013
                GROUP BY lat, long, date, gmt, src
                ORDER BY totals DESC
                LIMIT 10;''')
cursor.fetchall()

cursor.execute('''WITH totals_table AS 
                (SELECT COUNT(*) as totals, lat, long, date, gmt, src
                FROM merged_2013
                GROUP BY lat, long, date, gmt, src)
                
                SELECT lat, long, date, gmt, src 
                FROM totals_table 
                WHERE totals > 1;''')
duplicates_list = cursor.fetchall()

# Let's just go down the above list and figure out what is going on. 
duplicates_info = []
for duplicate in duplicates_list[:20]: 
    lat_coord, long_coord, date1, gmt, src = duplicate
    
    cursor.execute('''SELECT fire_name, fire, unit_id, agency
                    FROM merged_2013
                    WHERE lat = {} and long = {}
                        and gmt = {} and date = '{}'
                        and src = '{}'; '''.format(lat_coord, long_coord, gmt, date1, src))
    duplicates_info.append([cursor.fetchall(), duplicate])

for duplicate in duplicates_info[:10]: 
    print '-' * 50 
    print duplicate[0]
    print '\n' * 2
    print duplicate[1]



