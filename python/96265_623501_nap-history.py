import io
import pathlib
import logging

import dateutil.parser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import seaborn

import pyproj
import requests
import rtree


get_ipython().magic('matplotlib inline')

# we do this analysis for the main tide gauges
ids = ['DELFZL', 'DENHDR', 'HARLGN', 'HOEKVHLD', 'IJMDBTHVN', 'VLISSGN']

# datasets
ddl_url = 'https://waterwebservices.rijkswaterstaat.nl/METADATASERVICES_DBO/OphalenCatalogus/'

# Get a list of station info from the DDL
request = {
    "CatalogusFilter": {
        "Eenheden": True,
        "Grootheden": True,
        "Hoedanigheden": True
    }
}
resp = requests.post(ddl_url, json=request)
result = resp.json()

df = pd.DataFrame.from_dict(result['LocatieLijst'])
df = df.set_index('Code')
# note that there are two stations for IJmuiden. 
# The station was moved from the sluices to outside of the harbor in 1981.

# make a copy so we can add things
stations_df = df.loc[ids].copy()

# reproject to other coordinate systems

# We use pyproj
wgs84 = pyproj.Proj(init='epsg:4326')
rd = pyproj.Proj(init='epsg:28992')
etrs89_utm31n = pyproj.Proj(init='epsg:25831')

# compute coordinates in different coordinate systems
stations_df['x_rd'], stations_df['y_rd'] = pyproj.transform(
    etrs89_utm31n, 
    rd, 
    list(stations_df.X), 
    list(stations_df.Y)
)
stations_df['lon'], stations_df['lat'] = pyproj.transform(
    etrs89_utm31n, 
    wgs84, 
    list(stations_df.X), 
    list(stations_df.Y)
)

stations_df

# this is the file that was delivered by Rijkswaterstaat
# The file is a printed table, probably an extract from a database 
path = pathlib.Path('../../../data/rws/nap/historie/NAP_Historie.txt')
# open the file
stream = path.open()
# print the first few lines
print("".join(stream.readlines()[:5]))

# because the table is not in a standardized format, we have some cleaning up to do
lines = []
for i, line in enumerate(path.open()):
    # split the first and third line (only dashes and +)
    if i in (0, 2):
        continue
    # strip the | and whitespace
    fields = line.split('|')
    # put all fields in a list and strip of reamining | 
    fields = [field.strip().strip('|') for field in fields]
    # remove first and last column (should be empty)
    assert fields[0] == '' and fields[-1] == ''
    fields = fields[1:-1]
    # rejoin with the | (some fields contain , and ;, so we separate by |)
    line = "|".join(fields)
    # keep a list
    lines.append(line)
# concatenate cleaned up fields
txt = "\n".join(lines)

# read the CSV file as a table
df = pd.read_csv(io.StringIO(txt), sep='|', dtype={
    'titel': str,
    'x': float,
    'y': float
})
# make sure all titles are strings (some floats)
df['titel'] = df['titel'].astype(str)
# convert dates to dates
# did not check if the date format is consistent (not a common date format), let the parser guess
df['date'] = df['datum'].apply(lambda x: dateutil.parser.parse(x))

# based on the instructions, everything with an equal sign or after 2005 should be the revised NAP
# TODO: check if NAP correction is consistent with correction in use by PSMSL to create a local tide gauge benchmark
def is_revised(row):
    if row['date'].year >= 2005:
        return True
    if '=' in row['project_id']:
        return True
    return False
df['revised'] = df.apply(is_revised, axis=1)

# show the first few records
df.head()

# Some records don't have a location. We can't use them, so we'll drop them out of the dataset
total_n = len(df)
missing_n = total_n - len(df.dropna())
assert missing_n == df['x'].isna().sum(), "we expected all missings to be missings without coordinates"
logging.warning("%s records are dropped from the %s records", missing_n, total_n)
             
df = df.dropna()


fig, ax = plt.subplots(figsize=(10, 13))
ax.axis('equal')
ax.set_title('NAP benchmark history coverage')
ax.plot(df['x'], df['y'], 'g.', alpha=0.1, markersize=1, label='NAP measurement')
ax.plot(stations_df['x_rd'], stations_df['y_rd'], 'k.', label='main tide gauge')
for name, row in stations_df.iterrows():
    ax.annotate(xy=row[['x_rd', 'y_rd']], s=name)
ax.set_xlabel('x [m] EPSG:28992')
ax.set_ylabel('y [m] EPSG:28992')
ax.legend(loc='best');

# create a list of all NAP marks
grouped = df.groupby('ID')
nap_marks = grouped.agg('first')[['x', 'y']]
nap_marks.head()

# create a rtree to be able to quickly lookup nearest points
index = rtree.Rtree(
    (i, tuple(row) + tuple(row), obj)
    for i, (obj, row)
    in enumerate(nap_marks.iterrows())
)

# here we'll create a list of records that are close to our tide gauges
closest_dfs = []
for station_id in ids:
    # our current station
    station = stations_df.loc[station_id]
    # benchmarks near our current station
    nearest_ids = list(
        item.object
        for item 
        in index.nearest(
            tuple(station[['x_rd', 'y_rd']]), 
            num_results=10000, 
            objects=True
        )
    )
    # lookup corresponding records
    closest = df[np.in1d(df.ID, nearest_ids)].copy()
    # compute the distance
    closest['distance'] = np.sqrt((station['x_rd'] - closest['x'])**2 + (station['y_rd'] - closest['y'])**2)
    # set the station
    closest['station'] = station_id
    # drop all records further than 3km
    # you might want to use a more geological sound approach here, taking into account  faults, etc...
    closest = closest[closest['distance'] <= 3000]
    # sort
    closest = closest.sort_values(by=['station', 'ID', 'date'])
    # loop over each mark to add the 0 reference
    # the 0 reference is the height of the benchmark when it was first measured
    for it, df_group in iter(closest.groupby(('ID', 'x', 'y'))):
        df_group = df_group.copy()
        # add the 0 first height
        hoogtes = df_group['NAP hoogte']
        hoogtes_0 = hoogtes - hoogtes.iloc[0]
        df_group['hoogtes_0'] = hoogtes_0
        closest_dfs.append(df_group)

# combine all the selected records
selected = pd.concat(closest_dfs)
# problem with plotting dates correctly in seaborn, so we'll just use numbers
selected['year'] = selected['date'].apply(lambda x: x.year + x.dayofyear/366.0)

# show the number of measurements for each tide gauge
selected.groupby('station').agg('count')[['ID']]

palette = seaborn.palettes.cubehelix_palette(reverse=True)
cmap = seaborn.palettes.cubehelix_palette(reverse=True, as_cmap=True)
grid = seaborn.FacetGrid(selected, col="station", hue="distance", col_wrap=3, palette=palette, size=5)
grid.map(plt.plot, "year", "hoogtes_0", marker="o", ms=4, alpha=0.3)
grid.set(ylim=(-0.5, 0.1))
grid.set_ylabels('Height [m] related to 0 reference')
grid.set_xlabels('Time');
cbar_ax = grid.fig.add_axes([1.01, .3, .02, .4])  # <-- Create a colorbar axes
cb = matplotlib.colorbar.ColorbarBase(
    cbar_ax, 
    cmap=cmap,
    norm=matplotlib.colors.Normalize(0, 3000),
    orientation='vertical'
)



