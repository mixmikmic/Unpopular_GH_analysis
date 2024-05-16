import csv
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

# Created by Abhishek Kapoor
# Test.csv, is a custom generated file with only, Latitutes, Longitudes, Nkills Columns
filename = '/Users/abhishekkapoor/Desktop/Test.csv'

# For Ploting, Empty lists for Latitudes and Longitudes
ls, lsf, lo, lof = [], [], [], []
kills = []

# Reading the file
with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        ls.append(row[0])
        lo.append(row[1])
        kills.append(float(row[2]))

#Converting to Float        
lsf = [float(x) for x in ls]
lof = [float(x) for x in lo]

#Map Size
plt.figure(figsize=(16,12))

#Making the Map
map = Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0, lon_0=0)
map.shadedrelief()
map.drawcoastlines()
map.drawcountries()

plt.title("Scatter Plot as per the Number of Incidents since 1970")

#Converting Coordinates
x,y = map(lof, lsf)
map.plot(x, y, 'ro', markersize=4)

plt.show()

plt.figure(figsize=(16,12))

#Function to define the Colors associated with Number of Deaths
def mark_color(death):
    # Yellow for <10, Blue for <=50, Red for >50
    if death <= 10.0:
        return ('yo')
    elif death <= 50.0:
        return ('bo')
    else:
        return ('ro')

map = Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0, lon_0=0)
map.shadedrelief()
map.drawcoastlines()
map.drawcountries()

plt.title("Scatter Plot as per the Number of Deaths\n Yellow<=10, 10<Blue<=50, Red>50")

#Ploting Points
for long, lati, kill in zip(lof, lsf, kills):
    if kill == 0.0:
        min_mark = 0.0
    elif kill <= 10.0:
        min_mark = 2.0
    elif kill <= 50.0:
        min_mark = 0.5
    else:
        min_mark = 0.05
    x,y = map(long, lati)
    marker_size = kill * min_mark
    marker_color = mark_color(kill)
    map.plot(x, y, marker_color, markersize=marker_size)
 
plt.show()

import csv
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

# Created by Abhishek Kapoor
# Test.csv, is a custom generated file with only, Latitutes, Longitudes, Nkills Columns
filename = '/Users/abhishekkapoor/Desktop/Test.csv'

# For Ploting, Empty lists for Latitudes and Longitudes
ls, lsf, lo, lof = [], [], [], []
kills = []

# Reading the file
with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        ls.append(row[0])
        lo.append(row[1])
        kills.append(float(row[2]))

#Converting to Float        
lsf = [float(x) for x in ls]
lof = [float(x) for x in lo]

plt.figure(figsize=(16,12))

map = Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0, lon_0=0)
map.drawcoastlines(color='lightblue')
map.drawcountries(color='lightblue')
map.fillcontinents()
map.drawmapboundary()

plt.title("Heat Map as per the Number of Incidents since 1970")
   
x,y = map(lof, lsf)
map.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#424FA4',markeredgecolor="none", alpha=0.13)

 
plt.show()

import csv
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

# Created by Abhishek Kapoor
# Test.csv, is a custom generated file with only, Latitutes, Longitudes, Nkills Columns
filename = '/Users/abhishekkapoor/Desktop/Test.csv'

# For Ploting, Empty lists for Latitudes and Longitudes
ls, lsf, lo, lof = [], [], [], []
kills = []

# Reading the file
with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        ls.append(row[0])
        lo.append(row[1])
        kills.append(float(row[2]))

#Converting to Float        
lsf = [float(x) for x in ls]
lof = [float(x) for x in lo]

plt.figure(figsize=(16,12))

map = Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0, lon_0=0)
map.drawcoastlines(color='lightblue')
map.drawcountries(color='lightblue')
map.fillcontinents()
map.drawmapboundary()

plt.title("Heat Map as per the Number of Deaths")

for long, lati, kill in zip(lof, lsf, kills):
    if kill == 0.0:
        mcolor = '#ADD8E6'
        zord = 0
    elif kill <= 5.0:
        mcolor = '#80a442'
        zord = 2
    elif kill <= 30.0:
        mcolor = '#424fa4'
        zord = 4
    else:
        mcolor = '#a46642'
        zord = 6
    x,y = map(long, lati)
    map.plot(x, y, 'o', markersize=5,zorder=zord, markerfacecolor=mcolor, markeredgecolor="none", alpha=0.13)

 
plt.show()



