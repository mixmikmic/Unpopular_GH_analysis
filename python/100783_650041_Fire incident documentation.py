import numpy as np
import pandas as pd
from datetime import datetime

path = 'C:\\Users\\Kevin\\Desktop\\Fire Risk\\Model_matched_to_EAS'
incident_df = pd.read_csv(path + '\\' + 'matched_Fire_Incidents.csv', 
              low_memory=False)[['Incident Date','Primary Situation','EAS']].dropna()  #Drop obs  where any variable NAN

incident_df['Incident Date'] = pd.to_datetime(incident_df['Incident Date'])
incident_df['Incident_Year'] = incident_df['Incident Date'].dt.year

incident_df.head()

incident_df['code'] = incident_df['Primary Situation'].apply(lambda s: s[0:3])
pd.set_option("display.max_rows",999)
incident_df.groupby(['Primary Situation', 'code']).count()

di = {'FIRE OTHER': ['1 -', '10', '100', '11', '112'], 
      'BUILDING FIRE': ['111'], 
      'COOKING FIRE': ['113'], 
      'TRASH FIRE (INDOOR)': ['114','115','116','117','118'],
      'VEHICLE FIRE': ['120', '121', '122', '123', '130', '131', '132', '133', '134', '135', '136', '137', '138'],
      'OUTDOOR FIRE': ['140', '141', '142', '143', '150', '151', '152', '153', '154', '155', '160', '161', '162', '163', '164', '170', '173']}
# reverse the mapping
di = {d:c for c, d_list in di.items()
        for d in d_list}
#Map to 'Incident_Cat' groupings var
incident_df['Incident_Cat'] = incident_df['code'].map(di)

incident_df['Incident_Cat'].value_counts()

incident_df['Incident_Dummy'] = 1
incident_df = incident_df[['Incident Date', 
                           'EAS', 
                           'Incident_Year', 
                           'Incident_Cat', 
                           'Incident_Dummy']] 

incident_df.head()

#Export data
incident_df.to_csv(path_or_buf= path + '\\' + 'fireincident_data_formerge_20170917.csv', index=False)



