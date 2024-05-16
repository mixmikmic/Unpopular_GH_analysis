#Converts the csv file into a list of lists
import csv
file = open("guns.csv", "r")
temp = csv.reader(file)
data = list(temp)

data[0:3]

headers = data[0:1]
data = data[1:]
print(headers)
print("-------------------------------------")
print(data[0:5])

#Extracts the 'year' column from the list of lists
years = [row[1] for row in data]
print(years[0:10])

year_counts = {}
for i in years:
    if i in year_counts:
        year_counts[i] += 1
    else:
        year_counts[i] = 1
year_counts

import datetime

#The day is not specified in our data, this value will be assignedd as 1.
dates = [datetime.datetime(year=int(row[1]), month=int(row[2]), day=1) for row in data]
date_counts = {}
for i in dates:
    if i in date_counts:
        date_counts[i] += 1
    else:
        date_counts[i] = 1
date_counts

sex = [row[5] for row in data]
sex_counts = {}
for i in sex:
    if i in sex_counts:
        sex_counts[i] += 1
    else:
        sex_counts[i] = 1
print(sex_counts)

race = [row[7] for row in data]
race_counts = {}
for i in race:
    if i in race_counts:
        race_counts[i] += 1
    else:
        race_counts[i] = 1
race_counts

f2 = open("census.csv", "r")
temp2 = csv.reader(f2)
census = list(temp2)
census[0:2]

mapping = {}
mapping['Asian/Pacific Islander'] = 674625+6984195
mapping['Black'] = 44618105
mapping['Hispanic'] = 44618105
mapping['Native American/Native Alaskan'] = 15159516
mapping['White'] = 197318956

race_per_hundredk = {}
#We can iterate both the key and the value in a dictionary using .items()
for key, value in (race_counts.items()):    
    race_per_hundredk[key] = value*100000/mapping[key]
race_per_hundredk

intent = [row[3] for row in data]

races = [row[7] for row in data]
homicide_race_counts = {}
#We can iterate over both the index and the element in the list using the enumerate() function
for i, rac in enumerate(races):
    if intent[i] == 'Homicide':
        if rac in homicide_race_counts:
            homicide_race_counts[rac] += 1
        else:
            homicide_race_counts[rac] = 1
            
homicide_race_counts

homicide_race_per_hundredk = {}
for key, value in (homicide_race_counts.items()):    
    homicide_race_per_hundredk[key] = value*100000/mapping[key]
homicide_race_per_hundredk



