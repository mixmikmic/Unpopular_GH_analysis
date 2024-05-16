import math

origin = (42.9446,-122.1090)
radius = 10 # miles

latitude = origin[0]
longitude = origin[1]

lat_plus = latitude + radius/69  #69 miles / 1deg latitude
lat_minus = latitude - radius/69
long_plus = longitude + radius/(69.17 * math.cos(lat_plus))
long_minus = longitude - radius/(69.17 * math.cos(lat_minus))

[lat_plus, lat_minus, long_plus, long_minus]

query = "select facilityname, facilitylatitude, facilitylongitude, sites_available, cg_fcfs, cg_flush, cg_shower, cg_vault from daily where facilitylatitude between " + str(lat_minus) + " and " + str(lat_plus) + " and facilitylongitude between " + str(long_minus) + " and " + str(long_plus) + ";"

query



