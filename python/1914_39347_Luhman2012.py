import warnings
warnings.filterwarnings("ignore")

from astropy.io import ascii

import pandas as pd

tbl1 = ascii.read("http://iopscience.iop.org/0004-637X/758/1/31/suppdata/apj443828t1_mrt.txt")

tbl1.columns

tbl1[0:5]

len(tbl1)

from astroquery.simbad import Simbad
import astropy.coordinates as coord
import astropy.units as u

customSimbad = Simbad()
customSimbad.add_votable_fields('otype', 'sptype')

query_list = tbl1["Name"].data.data
result = customSimbad.query_objects(query_list, verbose=True)

result[0:3]

print "There were {} sources queried, and {} sources found.".format(len(query_list), len(result))
if len(query_list) == len(result):
    print "Hooray!  Everything matched"
else:
    print "Which ones were not found?"

def add_input_column_to_simbad_result(self, input_list, verbose=False):
    """
    Adds 'INPUT' column to the result of a Simbad query

    Parameters
    ----------
    object_names : sequence of strs
            names of objects from most recent query
    verbose : boolean, optional
        When `True`, verbose output is printed

    Returns
    -------
    table : `~astropy.table.Table`
        Query results table
    """
    error_string = self.last_parsed_result.error_raw
    fails = []

    for error in error_string.split("\n"):
        start_loc = error.rfind(":")+2
        fail = error[start_loc:]
        fails.append(fail)

    successes = [s for s in input_list if s not in fails]
    if verbose:
        out_message = "There were {} successful Simbad matches and {} failures."
        print out_message.format(len(successes), len(fails))

    self.last_parsed_result.table["INPUT"] = successes

    return self.last_parsed_result.table

result_fix = add_input_column_to_simbad_result(customSimbad, query_list, verbose=True)

tbl1_pd = tbl1.to_pandas()
result_pd = result_fix.to_pandas()
tbl1_plusSimbad = pd.merge(tbl1_pd, result_pd, how="left", left_on="Name", right_on="INPUT")

tbl1_plusSimbad.head()

get_ipython().system(' mkdir ../data/Luhman2012/')

tbl1_plusSimbad.to_csv("../data/Luhman2012/tbl1_plusSimbad.csv", index=False)

