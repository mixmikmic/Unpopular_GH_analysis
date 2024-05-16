get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import os; os.chdir('..')
import models
import orca
from urbansim.maps import dframe_explorer

# Run some model steps
orca.run([
    "neighborhood_vars",
    "rsh_simulate",
    "rrh_simulate",    
], iter_vars=[2010])

d = {tbl: orca.get_table(tbl).to_frame() for tbl in 
         ['buildings', 'residential_units', 'households']}

dframe_explorer.start(d, 
        center=[37.7792, -122.2191],
        zoom=11,
        shape_json='data/zones.json',
        geom_name='ZONE_ID', # from JSON file
        join_name='zone_id', # from data frames
        precision=2)



