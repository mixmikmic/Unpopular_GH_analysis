#import required Libraries
import pandas as pd
import numpy as np

#Get school level data
url="https://raw.githubusercontent.com/jakemdrew/EducationDataNC/master/Raw%20Datasets/All_Data_By_School_Final.csv"
schData=pd.read_csv(url, low_memory=False, dtype={'unit_code': object})

#Get school level test scores data 
url="https://raw.githubusercontent.com/jakemdrew/EducationDataNC/master/Raw%20Datasets/1516_Test_Scores.csv"
testScores=pd.read_csv(url, low_memory=False, dtype={'unit_code': object})

#Get school level racial compositions data
url="https://raw.githubusercontent.com/jakemdrew/EducationDataNC/master/Raw%20Datasets/" +      "Ec_Pupils_Expanded%20(2017%20Race%20Compositions%20by%20School).csv"
raceComps=pd.read_csv(url, low_memory=False, dtype={'unit_code': object, 'LEA': object})

#Convert our primary key to the proper data type before joining. 
raceComps.unit_code = raceComps.unit_code.astype('object')

#Review dataset contents before merging
print('****************School Data*********************************')
schData.info(verbose=False)
print('****************Test Scores*********************************')
testScores.info(verbose=False)
print('****************Racial Compostions**************************')
raceComps.info(verbose=False)


#Merge schoolData and testScores into a single file using school / unit code
schoolData = schData.merge(testScores, on='unit_code', how='left', suffixes=('_schoolData', '_testScores'))

#Review dataset contents after merging
print('****************After testScores Merge**********************')
schoolData.info(verbose=False)

#Merge schoolData and raceComps into a single file using school / unit code
schoolData = schoolData.merge(raceComps, on='unit_code', how='left', suffixes=('', '_Drop'))
#Remove any duplicate columns from racial compostion data file
dropCols = [x for x in schoolData.columns if x.endswith('_Drop')]
schoolData = schoolData.drop(dropCols, axis=1)

#Review dataset contents after merging
print('****************After raceComps Merge***********************')
schoolData.info(verbose=False)

# Map flag fields into bool or categorial  
schoolData['title1_type_flg'] = schoolData['title1_type_flg'].map({-1:True, 0:False})
schoolData['clp_ind_flg'] = schoolData['clp_ind_flg'].map({-1:True, 0:False})
schoolData['focus_clp_flg'] = schoolData['focus_clp_flg'].map({-1:True, 0:False})
schoolData['summer_program_flg'] = schoolData['summer_program_flg'].map({-1:True, 0:False})
schoolData['asm_no_spg_flg'] = schoolData['asm_no_spg_flg'].map({-1:True, 0:False})
schoolData['no_data_spg_flg'] = schoolData['no_data_spg_flg'].map({-1:True, 0:False})
schoolData['stem_flg'] = schoolData['stem_flg'].map({-1:True, 0:False})
schoolData['esea_status'] = schoolData['esea_status'].map({'P':'Esea_Pass', 'F':'Esea_Fail', np.nan:'Non_Esea'})
schoolData['Grad_project_status'] = schoolData['Grad_project_status'].map({'Y':True, 'N':False, np.nan:False})

#Save the indexs for records with a district missing
missingLEAs = schoolData[schoolData['LEA'].isna() == True].index

#Update the district to be the first 2 or 3 digits of the unit_code
schoolData.loc[schoolData['LEA'].isna() == True, 'LEA'] = schoolData['unit_code'].transform(lambda x: str(x[:-3]))

#Check that our update worked as expected
#schoolData.loc[missingLEAs][['LEA','unit_code']]

raceCompositionFields = [ 'Indian Male','Indian Female','Asian Male','Asian Female'
                         ,'Hispanic Male','Hispanic Female','Black Male','Black Female'
                         ,'White Male','White Female','Pacific Island Male','Pacific Island Female'
                         ,'Two or  More Male','Two or  More Female','Total','White','Black','Hispanic'
                         ,'Indian','Asian','Pacific Island','Two or More','White_Pct','Majority_Minority']

#Save the indexs for records with race compisitions missing
missingRace = schoolData[schoolData[raceCompositionFields].isna() == True].index

#Update missing race values with the district average when avaiable 
schoolData[raceCompositionFields] = schoolData.groupby('LEA')[raceCompositionFields].transform(
                                          lambda x: x.fillna(x.mean()))

#Check that our update worked as expected
#schoolData.loc[missingRace][schoolData['Indian Male'].isna()][['LEA','Indian Male']]

schoolData = schoolData [((schoolData.category_cd == 'M') | 
                          (schoolData.category_cd == 'T') | 
                          (schoolData.category_cd == 'A') |
                          (schoolData.category_cd == 'I')) &
                          (schoolData.student_num > 0) & 
                          (schoolData.type_cd_txt == 'Public') & 
                          (schoolData.school_type_txt == 'Regular School')
                         ]

schoolData.info()

#Remove fields not needed for machine learning
excludeFields = ['unit_code', 'Year', 'street_ad','scity_ad'
                 ,'state_ad','szip_ad','District Name','School Name','SBE District'
                 ,'grades_BYOD','grades_1_to_1_access'
                #raceComp fields to drop
                 ,'LEA','School','___School Name___','____LEA Name____'
                ]

schoolData = schoolData.drop(excludeFields,axis=1)

#Review dataset contents after drops
schoolData.info()
print('Columns Deleted: ', len(excludeFields))

#Remove any fields that have the same value in all rows
UniqueValueCounts = schoolData.apply(pd.Series.nunique)
SingleValueCols = UniqueValueCounts[UniqueValueCounts == 1].index
schoolData = schoolData.drop(SingleValueCols, axis=1)

#Review dataset contents after drops
schoolData.info()
print('Columns Deleted: ', len(SingleValueCols))

#Remove any fields that have unique values in every rows
schoolDataRecordCt = schoolData.shape[0]
UniqueValueCounts = schoolData.apply(pd.Series.nunique)
AllUniqueValueCols = UniqueValueCounts[UniqueValueCounts == schoolDataRecordCt].index
schoolData = schoolData.drop(AllUniqueValueCols, axis=1)

#Review dataset contents after drops
schoolData.info()
print('Columns Deleted: ', len(AllUniqueValueCols))

#Remove any empty fields (null values in every row)
schoolDataRecordCt = schoolData.shape[0]
NullValueCounts = schoolData.isnull().sum()
NullValueCols = NullValueCounts[NullValueCounts == schoolDataRecordCt].index
schoolData = schoolData.drop(NullValueCols, axis=1)

#Review dataset contents after empty field drops
schoolData.info()
print('Columns Deleted: ', len(NullValueCols))

#Isolate continuous and categorical data types
#These are indexers into the schoolData dataframe and may be used similar to the schoolData dataframe 
sD_boolean = schoolData.loc[:, (schoolData.dtypes == bool) ]
sD_nominal = schoolData.loc[:, (schoolData.dtypes == object)]
sD_continuous = schoolData.loc[:, (schoolData.dtypes != bool) & (schoolData.dtypes != object)]
print "Boolean Columns: ", sD_boolean.shape[1]
print "Nominal Columns: ", sD_nominal.shape[1]
print "Continuous Columns: ", sD_continuous.shape[1]
print "Columns Accounted for: ", sD_nominal.shape[1] + sD_continuous.shape[1] + sD_boolean.shape[1]

#Eliminate continuous columns with more than missingThreshold percentage of missing values
missingThreshold = 0.65
schoolDataRecordCt = sD_continuous.shape[0]
missingValueLimit = schoolDataRecordCt * missingThreshold
NullValueCounts = sD_continuous.isnull().sum()
NullValueCols = NullValueCounts[NullValueCounts >= missingValueLimit].index
schoolData = schoolData.drop(NullValueCols, axis=1)

#Review dataset contents after empty field drops
schoolData.info()

#Isolate categorical variables
sD_nominal = schoolData.loc[:, (schoolData.dtypes == object)]
#one hot encode categorical variables
schoolData = pd.get_dummies(data=schoolData, 
                       columns=sD_nominal, drop_first=True)

#Review dataset contents after empty field drops
schoolData.info()

#Replace all remaining NaN with 0
schoolData = schoolData.fillna(0)

#Check for Missing values again 
missing_values = schoolData.isnull().sum().reset_index()
missing_values.columns = ['Variable Name', 'Number Missing Values']
missing_values = missing_values[missing_values['Number Missing Values'] > 0] 
missing_values

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# calculate the correlation matrix
corr_matrix  = schoolData.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# plot the heatmap
fig, ax = plt.subplots(figsize=(180,360)) 
sns.set(font_scale=8)
sns.heatmap(upper[to_drop] , linewidths=.15, ax=ax)

#Restore fontscale back to normal after heatmap
sns.set(font_scale=1)

#Print a list of the columns with correlations > .95
print("Columns to delete with greater than .95 correlation: ",  len(to_drop))
print("*****************************************")
to_drop

#Check columns before drop 
schoolData.info()

# Drop the highly correlated features from our training data 
schoolData = schoolData.drop(to_drop, axis=1)

#Check columns after drop 
print('*********************************************')
schoolData.info()

#Write the final dataset to a .csv file for later use!
file_path = "D:/BenepactLLC/Belk/NC_Report_Card_Data/February 2018 Report/Datasets/MiddleSchoolsML_02_2018_Expanded.csv"
schoolData.to_csv(file_path, sep=',', index=False)

# Find all the categorical variables
schoolData_Bool = schoolData.loc[:, schoolData.dtypes == bool]
schoolData_Vars = schoolData.loc[:, schoolData.dtypes == object]
schoolData_Ohe_Vars = schoolData.select_dtypes(include='uint8')
cat_list_obj = list(pd.concat([schoolData_Vars, schoolData_Bool, schoolData_Ohe_Vars]))
cat_list_obj_len = len(cat_list_obj)
# Examine categorical variables of interest  
import matplotlib.pyplot as plt

print('Total categorical columns: ', cat_list_obj_len)

for i in range(0,len(cat_list_obj)):
    plt.figure(figsize = (18,4))
    ax = schoolData[cat_list_obj[i]].value_counts().plot(kind='bar')
    plt.title(cat_list_obj[i])
    plt.show()

import sklearn
import pandas as pd

print('Sklearn Version: ' + sklearn.__version__)
print('Pandas Version: ' + pd.__version__)



