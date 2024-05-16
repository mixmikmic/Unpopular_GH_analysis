get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
if 'sim' not in globals():
    import os; os.chdir('..')
import models
import orca
import pandas as pd
pd.set_option('display.max_columns', 500)

cl = orca.get_table('craigslist').to_frame()
cl[1:5]

cl.describe()

get_ipython().run_cell_magic('capture', '', 'orca.run(["neighborhood_vars"])')

# The model expression is in rrh.yaml; price_per_sqft is the asking monthly rent per square 
# foot from the Craigslist listings. Price, sqft, and bedrooms are specific to the unit, 
# while all the other variables are aggregations at the node or zone level. Note that we 
# can't use bedrooms in the simulation stage because it's not in the unit data.

orca.run(["rrh_estimate"])

# to save variations, create a new yaml file and run this to register it

@orca.step()
def rh_cl_estimate_NEW(craigslist, aggregations):
    return utils.hedonic_estimate("rrh_NEW.yaml", craigslist, aggregations)

orca.run(["rrh_estimate_NEW"])

orca.run(["rsh_estimate"])









