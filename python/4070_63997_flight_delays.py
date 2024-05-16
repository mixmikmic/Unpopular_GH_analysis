import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

names = ['flight_date', 'airline_id', 'flight_num', 'origin', 'destination',
         'departure_time', 'departure_delay', 'arrival_time', 'arrival_delay', 'air_time', 'distance']

flights_raw = pd.read_csv('flights.csv', parse_dates=True, names=names)
flights_raw.head(3)

flights_raw.describe()

_ = plt.hist(flights_raw.arrival_delay, bins=50, range=[-100, 250])
plt.xlabel('Arrival Delay (minutes)')
plt.ylabel('Count')

airlines_raw = pd.read_csv('airlines.csv')
airlines_raw.head(3)

airports_raw = pd.read_csv('airports.csv')
airports_raw.head(3)

flights = flights_raw.merge(airlines_raw, left_on='airline_id', right_on='Code', how='inner')
flights.head(3)

avg_delay_by_airline = flights.groupby('Description').agg({'arrival_delay': [np.size, np.mean]})
avg_delay_by_airline.sort_values([('arrival_delay', 'mean')], ascending=True, inplace=True)

ints = [i for i in range(len(avg_delay_by_airline.index))]
plt.barh(ints, avg_delay_by_airline[('arrival_delay', 'mean')].values)
plt.yticks(ints, avg_delay_by_airline.index)
plt.xlabel('Arrival Delay (Minutes)')

