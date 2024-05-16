import pandas

PATH_TO_CSV = "./data/matched_Fire_Safety_Complaints.csv"
complaints = pandas.read_csv(PATH_TO_CSV)

from collections import Counter

complaint_types = Counter(complaints["Complaint Item Type Description"])
complaint_types

disposition_types = Counter(complaints["Disposition"])
disposition_types

from datetime import date

def is_valid_complaint(row):
    disposition = row["Disposition"]
    return not (disposition == "no merit" or disposition == "duplicate complaint")

def is_corrected(row):
    disposition = row["Disposition"]
    return disposition == "condition corrected"

def parse_date(date_str):
    """For a string in the format of YYYY-MM-DD, 
    return (YYYY, MM, DD)"""
    return tuple(map(int, date_str.split('-')))

def is_within_date_range(row, min_date_str, max_date_str):
    """checks if beg <= row["Received Date"] <= end
    row: a row in the dataset, representing one complaint
    min_date_str: a str representing the beginning of the date range
    max_date_str: a str representing the end of the date range
    """
    complaint_date = date(*parse_date(row["Received Date"]))
    min_date = date(*parse_date(min_date_str))
    max_date = date(*parse_date(max_date_str))
    
    return min_date <= complaint_date and max_date >= complaint_date

# get the mappting from Complaint Item Type Description to Complaint Item Type
complaint_id_mapping = {}

for i, r in complaints.iterrows():
    dsc = r["Complaint Item Type Description"]
    complaint_id = r["Complaint Item Type"]
    if dsc in complaint_id_mapping:
        if complaint_id_mapping[dsc] != complaint_id:
            raise Exception("Complaint Type has different IDs")
    else:
        complaint_id_mapping[dsc] = complaint_id

complaint_id_mapping

# define mapping from complaint item type to category
potential_fire_cause = "potential fire cause"
potential_fire_control = "potential fire control"
fire_emergency_safety = "fire emergency safety"
multiple_violations = "multiple violations"

complaint_category_mapping = {"potential fire cause":['15', '13', '18', '10', '01', '07', '12', '20', '04'],
                              "potential fire control":['05', '19', '06'], 
                              "fire emergency safety": ['03', '24', '22', '02', '23', '21', '11']}
# reverse the mapping to get id -> category mappings
complaint_category_mapping = {d:c for c, d_list in complaint_category_mapping.items()
                                  for d in d_list}

from collections import defaultdict
from math import isnan

eas_to_features = defaultdict(lambda :defaultdict(float))

for d, r in complaints.iterrows():
    eas = r["EAS"]
    complaint_type = r["Complaint Item Type"]
    if not isnan(eas) and is_within_date_range(r, "2005-01-01", "2016-12-31"):
        features = eas_to_features[int(eas)]
        # increment count features for generalized complaint types
        if complaint_type in complaint_category_mapping and is_valid_complaint(r):
            feature_name = "count {}".format(complaint_category_mapping[complaint_type])
            features[feature_name] += 1
            features["count all complaints"] += 1
        
            # increment count features for generalized complaint types not corrected:
            if not is_corrected(r):
                feature_name = "count {} not corrected".format(complaint_category_mapping[complaint_type])
                features[feature_name] += 1
                features["count all complaints not corrected"] += 1
        
        # count for each complaint type, maybe remove this?
        #complaint_type_dsc = r["Complaint Item Type Description"]
        #if is_valid_complaint(r):
        #    feature_name = "count {}".format(complaint_type_dsc)
        #    features[feature_name] += 1
        #    
        #    if not is_corrected(r):
        #        feature_name = "count {} not corrected".format(complaint_type_dsc)
        #        features[feature_name] += 1

df = pandas.DataFrame.from_dict(eas_to_features, orient='index', dtype=float)
df.fillna(0, inplace=True)
df

