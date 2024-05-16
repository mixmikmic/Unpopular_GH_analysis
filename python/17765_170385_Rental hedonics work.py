# Sam Maurer, June 2015

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import models
import urbansim.sim.simulation as sim
from urbansim.utils import misc

import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')

s = sim.get_injectable('store')
s

# Where does the data for hedonic estimation come from?
# In rsh.yaml, the model expression is: 
'''
np.log(price_per_sqft) ~ I(year_built < 1940) + I(year_built > 2000)
    + np.log1p(sqft_per_unit) + ave_income + stories + poor + renters + sfdu + autoPeakTotal
    + transitPeakTotal + autoOffPeakRetail + ave_lot_size_per_unit + sum_nonresidential_units
    + sum_residential_units
'''

s.costar.columns.values

s.homesales.columns.values

s.parcels.columns.values

s.buildings.columns.values

# Many of the inputs come from the neighborhood_vars model, which does network aggregation
# and stores its results in the 'nodes' table -- and others are defined in variables.py
'''
price_per_sqft:              homesales (which does not come from the h5 table, but is 
                                 constructed on the fly from the buildings table)
                                 buildings > redfin_sale_price and sqft_per_unit
                                 
year_built:                  buildings
sqft_per_unit:               buildings dynamic column
ave_income:                  nodes, from households > income
stories:                     buildings
poor:                        nodes, from households > persons
renters:                     nodes, from households > tenure
sfdu:                        nodes, from buildings > building_type_id
autoPeakTotal:               logsums
transitPeakTotal:            logsums
autoOffPeakRetail:           logsums
ave_lot_size_per_unit:       nodes, from buildings dynamic column
sum_nonresidential_units:    nodes, from buildings dynamic column
sum_residential_units:       nodes, from buildings > residential_units
'''

# Note for future -- best way to look at the data urbansim is actually using is to call 
# sim.get_table(), because the h5 file is only a starting point for the data structures

# Craigslist gives us x,y coordinates, but they're not accurate enough to link
# to a specific parcel. Probably the best approach is to set up a new table for CL
# data, and then use a broadcast to link them to the nodes and logsums tables

# This data is from Geoff Boeing, representing 2.5 months of listings for the SF Bay Area
# Craigslist region. He's already done some moderate cleaning, filtering for plausible
# lat/lon and square footage values and removing posts that were duplicated using CL's
# 'repost' tool. Other duplicate listings still remain.

df = pd.read_csv(os.path.join(misc.data_dir(), "sfbay_craigslist.csv"))

df.describe()

# Borrowing code from datasources.py to link x,y coods to nodes
net = sim.get_injectable('net')
df['_node_id'] = net.get_node_ids(df['longitude'], df['latitude'])

df['_node_id'].describe()

df.head(5)

# - Need to deal with NA's
# - Should also choose some outlier thresholds

df.isnull().sum()

df['bedrooms'] = df.bedrooms.replace(np.nan, 1)
df['neighborhood'] = df.neighborhood.replace(np.nan, '')

df.isnull().sum()

df.price_sqft[df.price_sqft<8].hist(bins=50, alpha=0.5)

# try 0.5 and 7 as thresholds to get rid of worst outliers









