import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import statsmodels.api as sm
import patsy

get_ipython().magic('matplotlib inline')

#Read in data
path = 'C:\\Users\\Kevin\\Desktop\\Fire Risk\\Model_matched_to_EAS'
complaint_df = pd.read_csv(path + '\\' + 'matched_Fire_Safety_Complaints.csv', 
              low_memory=False)
incident_df = pd.read_csv(path + '\\' + 'matched_Fire_Incidents.csv', 
              low_memory=False)

#Functions
def topcom(data, strvar):
    t = pd.DataFrame(data.groupby(strvar)[strvar].count().sort_values(ascending=False))
    t.rename(columns = {strvar: 'Count'}, inplace=True)
    t[strvar] = t.index
    return t

def my_barplot(xaxis, yaxis, dta, title):
    g1 = sns.barplot(x=xaxis, y=yaxis, data=dta, palette='coolwarm_r')
    g1.set_xticklabels(labels=dta[xaxis], rotation=90)
    g1.set_title(title)
    sns.despine()
    return g1

#Subset complaint data, keep only 2006-2016 to maintain comparability
complaint_df = complaint_df[['Complaint Item Type Description',
               'Received Date',
               'Disposition',
               'Neighborhood  District',  #Note two spaces b/w words
               'Closest Address',
               'EAS']].dropna(subset=['EAS'])  #Drop where EAS nan
           
complaint_df = complaint_df[(complaint_df['Received Date'] >= '2006-01-01') & 
              (complaint_df['Received Date'] <= '2016-12-31')]

               
#Subset fire incident data, keep only 2006-2016 to maintain comparability            
#Also, keep only building fire/cooking fire/trash fire                                    
incident_df = incident_df[['Incident Date',
                           'Closest Address',
                           'Primary Situation',
                           'Neighborhood  District',  #Note two spaces b/w words
                           'Property Use',
                           'EAS']].dropna(subset=['EAS'])  #Drop where EAS nan

incident_df.rename(columns = {'Primary Situation': 'Situation'}, inplace=True)

incident_df = incident_df[(incident_df.Situation.str.contains("111")) |
                          (incident_df.Situation.str.contains("113")) |
                          (incident_df.Situation.str.contains("118")) |
                          (incident_df.Situation.str.contains("150")) |
                          (incident_df.Situation.str.contains("151")) |
                          (incident_df.Situation.str.contains("154"))]

incident_df['Situation'] = incident_df['Situation'].str.replace('111 building fire', '111 - building fire')
incident_df['Situation'] = incident_df['Situation'].str.replace('113 cooking fire, confined to container', '113 - cooking fire, confined to container')
incident_df['Situation'] = incident_df['Situation'].str.replace('118 trash or rubbish fire, contained', '118 - trash or rubbish fire, contained')
incident_df['Situation'] = incident_df['Situation'].str.replace('150 outside rubbish fire, other', '150 - outside rubbish fire, other')
incident_df['Situation'] = incident_df['Situation'].str.replace('151 outside rubbish, trash or waste fire', '151 - outside rubbish, trash or waste fire')
incident_df['Situation'] = incident_df['Situation'].str.replace('154 dumpster or other outside trash receptacle fire', '154 - dumpster/outside trash receptacle fire')
                          
incident_df = incident_df[(incident_df['Incident Date'] >= '2006-01-01') & 
              (incident_df['Incident Date'] <= '2016-12-31')]

#Types of complaints
temp = topcom(complaint_df, 'Complaint Item Type Description')
my_barplot('Complaint Item Type Description', 'Count', temp , 'Types of Complaints, 2006-2016')

#Dispositions
temp = topcom(complaint_df, 'Disposition')
my_barplot('Disposition', 'Count', temp, 'Types of Complaint Dispositions, 2006-2016')

#Types of fire incidents (remember we removed a few of these)
temp = topcom(incident_df, 'Situation')
my_barplot('Situation', 'Count', temp, 'Types of Fire Incidents, 2006-2016')

#Top complaint districts
t1 = topcom(complaint_df, 'Neighborhood  District')
t1.rename(columns = {'Count': 'Complaint Count'}, inplace=True)
my_barplot('Neighborhood  District', 'Complaint Count', t1, 'Complaints by Neighborhood, 2006-2016')

#Top incident districts
t2 = topcom(incident_df, 'Neighborhood  District')
t2.rename(columns = {'Count': 'Incident Count'}, inplace=True)
my_barplot('Neighborhood  District', 'Incident Count', t2, 'Fire Incidents by Neighborhood, 2006-2016')

#Relationship b/w incidents and complaints
mrg_dta = pd.merge(t1, t2, on='Neighborhood  District')
sns.jointplot(x='Incident Count',y='Complaint Count',data=mrg_dta,kind='reg')

#Now, does this relationship hold at EAS level?  Lets see...
t3 = topcom(incident_df, 'EAS')
t3.rename(columns = {'Count': 'incident_count'}, inplace=True)
t3['incident_dummy'] = 1

t4 = topcom(complaint_df, 'EAS')
t4.rename(columns = {'Count': 'complaint_count'}, inplace=True)
t4['complaint_dummy'] = 1

mrg_dta2 = pd.merge(t3, t4, on='EAS', how='outer')
mrg_dta2 = mrg_dta2.fillna(0)
mrg_dta2.corr()

#Probit model fire incident dummy on number of complaints
f = 'incident_dummy ~ complaint_count'
incident_dummy, X = patsy.dmatrices(f, mrg_dta2, return_type='dataframe')
sm.Probit(incident_dummy, X).fit().summary()

