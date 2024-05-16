get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
if 'sim' not in globals():
    import os; os.chdir('..')
import models
import urbansim.sim.simulation as sim
from urbansim.maps import dframe_explorer

d = {tbl: sim.get_table(tbl).to_frame() for tbl in ['buildings', 'jobs', 'households']}

dframe_explorer.start(d, 
        center=[37.7792, -122.2191],
        zoom=11,
        shape_json='data/zones.json',
        geom_name='ZONE_ID', # from JSON file
        join_name='zone_id', # from data frames
        precision=2)



