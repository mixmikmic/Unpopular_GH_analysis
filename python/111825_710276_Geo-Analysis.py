import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
get_ipython().magic('matplotlib inline')

station_df = pd.read_csv('Data/StationEntrances.csv')
station_df.head(3)

top20_to_lat_long = {
 'GRDCNTRL42ST4567S': ('Grand Central - 42nd St', 40.751776, -73.976848),
 '34STHERALDSQBDFMNQR' : ('34 St - Herald Sq', 40.749567, -73.98795),
 'TIMESSQ42ST1237ACENQRS': ('Times Square - 42nd St', 40.754672, -73.986754),
 '14STUNIONSQ456LNQR': ('14 St - Union Sq', 40.735736, -73.990568),
 'FULTONST2345ACJZ': ('Fulton St', 40.710374, -74.007582),
 '34STPENNSTAACE': ('34 St - Penn Station (A C E)', 40.752287, -73.993391),
 '42STPORTAUTH1237ACENQRS': ('42 St - Port Authority Bus Terminal', 40.757308, -73.989735),
 '59STCOLUMBUS1ABCD': ('59 St - Columbus Circle', 40.768296, -73.981736),
 '59ST456NQR': ('59 St - Lexington Ave', 40.762526,-73.967967),
 '86ST456': ('86 St', 40.779492, -73.955589),
 '4750STSROCKBDFM': ('47 - 50 Sts - Rockefeller Ctr', 40.758663, -73.981329),
 'FLUSHINGMAIN7': ('Flushing - Main St', 40.7596, -73.83003),
 '34STPENNSTA123ACE': ('34 St - Penn Station (1 2 3)', 40.750373, -73.991057),
 'JKSNHTROOSVLT7EFMR': ('Jackson Hts - Roosevelt Ave', 40.746644, -73.891338),
 '42STBRYANTPK7BDFM': ('42 St - Bryant Pk', 40.754222, -73.984569),
 'ATLAVBARCLAY2345BDNQR': ('Atlantic Ave - Barclays Center', 40.683666, -73.97881),
 'CANALST6JNQRZ': ('Canal St', 40.718092, -73.999892 ),
 'LEXINGTONAV536EM': ('Lexington Ave - 53 St', 40.757552, -73.969055),
 '96ST123': ('96 St', 40.793919, -73.972323),
 '14ST123FLM': ('14 St - 7th Ave', 40.737826, -74.000201),
}

top20_description = []
for station in top20_to_lat_long.items():
    name = station[1][0]
    lat = station[1][1]
    lon = station[1][2]
    top20_description.append({'station': name, 'location': (lat,lon)})

data_daily = pd.read_csv('Data/data_daily.csv')
date = '06/30/2016'
for i,station in enumerate(top20_to_lat_long.keys()):
    count = data_daily[(data_daily.DATE==date) & (data_daily.UN_STATION==station)].groupby(['UN_STATION', 'DATE'])['SCP'].value_counts().count()
    transits = int(data_daily[(data_daily.DATE==date) & (data_daily.UN_STATION==station)].groupby(['UN_STATION', 'DATE'])['TRANSITING'].sum())
    top20_description[i]['turnstiles'] = count
    top20_description[i]['transits'] = transits

top20_df = pd.DataFrame(top20_to_lat_long).T
station_latitude = list(top20_df.loc[:,1])
station_longitude = list(top20_df.loc[:,2])

income_df = pd.read_csv('Data/medianhouseholdincomecensustract.csv')
income_df.head(3)

#Median Household Income more than 100k
rich_families = income_df[(income_df['MHI']>100000) &                          ((income_df['COUNTY'] == 'New York County')                           | (income_df['COUNTY'] == 'Kings County')                           | (income_df['COUNTY'] == 'Bronx County')                           | (income_df['COUNTY'] == 'Queens County'))]
rich_families.LOCALNAME.value_counts().head(5)

rich_latitude = rich_families.INTPTLAT10
rich_longitude = rich_families.INTPTLON10

irs_df = pd.read_excel('Data/IRS_SOI_NY_2014.csv')
irs_df = irs_df.ix[10:]
irs_df.head(4)

#Cutting off deduction by only those in the $100k or more bracket
irs_df = irs_df[(irs_df['Size of adjusted gross income']=='$100,000 under $200,000')             | (irs_df['Size of adjusted gross income']=='$200,000 or more')]

irs_df.rename(columns={'Total itemized deductions': 'Total Itemized Deductions Amount'},inplace=True)
irs_df.rename(columns={'Unnamed: 57': 'Amount of AGI'},inplace=True)

itemized_deduction = irs_df.groupby('ZIP\ncode [1]')['Total Itemized Deductions Amount'].sum()
Amount_of_AGI = irs_df.groupby('ZIP\ncode [1]')['Amount of AGI'].sum()
deduction_df = pd.concat([itemized_deduction, Amount_of_AGI], axis=1)

ratio = np.true_divide(itemized_deduction,Amount_of_AGI)
generous_ratio = ratio[ratio[:] > 0.01]
generous_zip_codes = list(generous_ratio.index)
zip_to_lat_long = pd.read_csv('Data/Zip_to_Lat_Lon.txt')

generous_lat = []
generous_long = []
for zip_code in generous_zip_codes:
    generous_lat.append(float(zip_to_lat_long[zip_to_lat_long.ZIP == zip_code].LAT))
    generous_long.append(float(zip_to_lat_long[zip_to_lat_long.ZIP == zip_code].LNG))

from bokeh.io import output_notebook, show
from bokeh.models import ( GMapPlot, GMapOptions, ColumnDataSource,                           Circle, DataRange1d, PanTool,                           WheelZoomTool, BoxSelectTool
)

def geoplotting(latitude,longitude,title):
    output_notebook()

    map_options = GMapOptions(lat=40.764811, lng=-73.973347,                               map_type="roadmap", zoom=11)

    plot = GMapPlot(
        x_range=DataRange1d(), y_range=DataRange1d(), \
        map_options=map_options
    )
    
    plot.title.text = title

    plot.api_key = "AIzaSyBaExoC_xY6qKJ4TF3MkW78Hhidr32ZSzg"

    source = ColumnDataSource(
        data=dict(
            lat=latitude, #needs to be a list of latitude
            lon=longitude, #needs to be a corresponding list of long
        )
    )

    circle = Circle(x="lon", y="lat", size=15, fill_color="blue", fill_alpha=0.8, line_color=None)
    plot.add_glyph(source, circle)

    plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
    show(plot)


geoplotting(station_latitude,station_longitude,"Top Stations")

import gmaps
gmaps.configure(api_key="AIzaSyBaExoC_xY6qKJ4TF3MkW78Hhidr32ZSzg")
top20_locations = zip(station_latitude,station_longitude)
rich_locations = zip(rich_latitude,rich_longitude)
generous_locations = zip(generous_lat,generous_long)

fig = gmaps.figure()
top20_locations = [station["location"] for station in top20_description]
info_box_template = """
<dl>
<dt>Station</dt><dd>{station}</dd>
<dt># of Turnstiles</dt><dd>{turnstiles}</dd>
<dt>Daily Transits (6/30/2016)</dt><dd>{transits}</dd>
</dl>
"""
station_info = [info_box_template.format(**station) for station in top20_description]
marker_layer = gmaps.marker_layer(top20_locations, info_box_content=station_info)
fig = gmaps.figure()
fig.add_layer(marker_layer);fig

fig = gmaps.figure()
fig.add_layer(gmaps.heatmap_layer(rich_locations,point_radius = 15));fig

fig = gmaps.figure()
fig.add_layer(gmaps.heatmap_layer(generous_locations,point_radius = 40,max_intensity=1));fig



