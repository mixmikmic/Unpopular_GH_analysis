import os
import json


import matplotlib.pyplot as plt
import matplotlib.patches as patch
get_ipython().magic('matplotlib inline')


import pandas as pd
import numpy as np



get_ipython().run_cell_magic('time', '', 'df = pd.DataFrame.from_csv("./labeled.control.dump.csv")\ndf["time"] = pd.to_datetime(df.created_at)\ndf = df.set_index("time")')

path = "/Users/JasonLiu/dump/predicted/"
files = os.listdir(path)

df = pd.concat(map(pd.read_csv, [path+file for file in files[1:]]))
df["time"] = pd.to_datetime(df.time)

df["fp"] = df["prediction_alcohol_svc"] * df["prediction_firstperson_svc"]

col = ["prediction_firstperson_level_0", "prediction_firstperson_level_2", "prediction_firstperson_level_3"]
new_fp_cols = ["casual", "looking", "reflecting"]
for new_name, old_name in zip(new_fp_cols, col):
    df[new_name] = df[old_name] * df.prediction_alcohol_svc * df.prediction_firstperson_svc

states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

states_inverted = {v:k for (k,v) in states.items()}

def map2name(s):
    if "USA" in s:
        state = s[:-5]
        if state in states_inverted:
            return state
        else:
            return "<Other>"
    try:
        state_code = s[-2:]
        return states[state_code]
    except:
        return "<Other>"
    return "<Other>"

df["location"] = df.place_fullname.astype(str).apply(map2name)

alcohol_depend = pd.read_csv("./academic_data/alcohol_dependence.csv").set_index("State")

dep = alcohol_depend["18 or Older\rEstimate"].apply(lambda _: _[:4]).astype(float)

location = df.groupby("location")

dep_fp = location.agg({"fp":"mean"})

temp = dep_fp.join(dep)
temp.columns = ["predicted", "measured"]

import seaborn as sns

temp.sort("predicted")["predicted"].head()

temp.sort("measured")["measured"].head()



