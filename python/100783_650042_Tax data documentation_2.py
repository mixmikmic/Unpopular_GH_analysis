import numpy as np
import pandas as pd
from datetime import datetime

path = 'C:\\Users\\Kevin\\Desktop\\Fire Risk\\Model_matched_to_EAS'

#This will take a while to load.  Very large file...
tax_df = pd.read_csv(path + '\\' + 'matched_EAS_Tax_Data.csv', 
              low_memory=False)[[
                 'EAS BaseID',
                 'Neighborhoods - Analysis Boundaries',
                 'Property Class Code',
                 'Property_Class_Code_Desc',
                 'Location_y',
                 'Address',
                 'Year Property Built',
                 'Number of Bathrooms',
                 'Number of Bedrooms',
                 'Number of Rooms',
                 'Number of Stories',
                 'Number of Units',
                 'Percent of Ownership',
                 'Closed Roll Assessed Land Value',
                 'Property Area in Square Feet',
                 'Closed Roll Assessed Improvement Value'
                 ]].dropna()

#Create land value per square foot var
tax_df['landval_psqft'] = tax_df['Closed Roll Assessed Land Value'] / tax_df['Property Area in Square Feet']

tax_df.rename(columns = {'EAS BaseID': 'EAS'}, inplace=True)
tax_df.rename(columns = {'Neighborhoods - Analysis Boundaries': 'Neighborhood'}, inplace=True)

tax_df.head()

def removal(var, low, high):
    tax_df[(tax_df[var]<=low) & (tax_df[var]<=high)]
    return tax_df

#Remove if 0 stories, remove if > 30 stories
tax_df = removal('Number of Stories',1,30)

#Remove if landvalue/sq_foot = 1 or > 1000
tax_df = removal('landval_psqft',1,1000)

#Remove if num. bathrooms, bedrooms, extra rooms > 100
tax_df = removal('Number of Bathrooms',0,100)
tax_df = removal('Number of Bedrooms',0,100)
tax_df = removal('Number of Rooms',0,100)

#Remove if year_built < 1880 or > 2017
tax_df = removal('Year Property Built',1880,2017)

#Remove num units > 250
tax_df = removal('Number of Units',0,250)

#Remove percent ownership < 0, > 1
tax_df = removal('Percent of Ownership',0,1)

#Create Tot_rooms var
tax_df['Tot_Rooms'] = tax_df['Number of Bathrooms'] +                     tax_df['Number of Bedrooms']  +                     tax_df['Number of Rooms']
        
#Subset to numeric vars only, group by EAS average          
tax_df_num = tax_df[[
                 'EAS',
                 'Year Property Built',
                 'Number of Bathrooms',
                 'Number of Bedrooms',
                 'Number of Rooms',
                 'Number of Stories',
                 'Number of Units',
                 'Percent of Ownership',
                 'Closed Roll Assessed Land Value',
                 'Property Area in Square Feet',
                 'Closed Roll Assessed Improvement Value',
                 'Tot_Rooms',
                 'landval_psqft'
                 ]].groupby(by='EAS').mean().reset_index()

pd.options.display.float_format = '{:.2f}'.format
tax_df_num.describe()

tax_df_str = tax_df[[
                 'EAS',
                 'Neighborhood',
                 'Property Class Code',
                 'Property_Class_Code_Desc',
                 'Location_y',
                 'Address',
                 ]].groupby(by='EAS').max().reset_index()

tax_df_str['Property_Class_Code_Desc'] = tax_df_str['Property_Class_Code_Desc'].apply(lambda x: x.upper())
tax_df_str['Neighborhood'] = tax_df_str['Neighborhood'].apply(lambda x: x.upper())

tax_df_str.head()

pd.set_option("display.max_rows",999)
tax_df_str.groupby(['Property Class Code', 'Property_Class_Code_Desc']).count()

di = {'APARTMENT': ['A', 'AC', 'DA', 'TIA'], 
      'DWELLING': ['D'], 
      'FLATS AND DUPLEX': ['F','F2','FA','TIF'], 
      'CONDO, ETC.': ['Z'],
      'COMMERCIAL USE': ['C','CD','B','C1','CD','CM','CZ'],
      'INDUSTRIAL USE': ['I','IDC','IW','IX','IZ'],
      'OFFICE' : ['O', 'OA','OAH', 'OBH', 'OBM', 'OC', 'OCH', 'OCL', 'OCM', 'OMD', 'OZ']}

# reverse the mapping
di = {d:c for c, d_list in di.items()
        for d in d_list}

#Map to 'Building_Cat' groupings var
tax_df_str['Building_Cat'] = tax_df_str['Property Class Code'].map(di)

#Remainders placed in "OTHER" category
x = ['APARTMENT', 'DWELLING', 'FLATS AND DUPLEX', 'CONDO, ETC.', 'COMMERCIAL USE', 'INDUSTRIAL USE', 'OFFICE']
tax_df_str.loc[~tax_df_str['Building_Cat'].isin(x), 'Building_Cat'] = 'OTHER'

tax_df_str['Building_Cat'].value_counts()

exp_df = pd.merge(tax_df_str, tax_df_num, how='left', on='EAS')
exp_df.drop(['Property Class Code', 'Property_Class_Code_Desc'], inplace=True, axis=1)

#Rename
exp_df.rename(columns = {'Year Property Built': 'Yr_Property_Built'}, inplace=True)
exp_df.rename(columns = {'Number of Bathrooms': 'Num_Bathrooms'}, inplace=True)
exp_df.rename(columns = {'Number of Bedrooms': 'Num_Bedrooms'}, inplace=True)
exp_df.rename(columns = {'Number of Rooms': 'Num_Rooms'}, inplace=True)
exp_df.rename(columns = {'Number of Stories': 'Num_Stories'}, inplace=True)
exp_df.rename(columns = {'Number of Units': 'Num_Units'}, inplace=True)
exp_df.rename(columns = {'Percent of Ownership': 'Perc_Ownership'}, inplace=True)
exp_df.rename(columns = {'Closed Roll Assessed Land Value': 'Land_Value'}, inplace=True)
exp_df.rename(columns = {'Property Area in Square Feet': 'Property_Area'}, inplace=True)
exp_df.rename(columns = {'Closed Roll Assessed Improvement Value': 'Assessed_Improvement_Val'}, inplace=True)

exp_df.info()

#Export data
exp_df.to_csv(path_or_buf= path + '\\' + 'tax_data_formerge_20170917.csv', index=False)



