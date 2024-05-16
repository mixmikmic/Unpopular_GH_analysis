#import required Libraries
import pandas as pd
import numpy as np
import os
import string

#**********************************************************************************
# Set the following variables before running this code!!!
#**********************************************************************************
#All raw data files are processed for the year below
schoolYear = 2017

#Location where copies of the raw data files will be read in from csv files.
dataDir = 'C:/Users/Jake/Documents/GitHub/EducationDataNC/' + str(schoolYear) + '/School Datasets/'

#Name of the file to be processed
#inputFileName = 'PublicSchools2017'
#inputFileName = 'PublicHighSchools2017'
#inputFileName = 'PublicMiddleSchools2017'
inputFileName = 'PublicElementarySchools2017'

#Input file being transformed for machine learning 
inputFile = dataDir + inputFileName + '.csv'

#Location where the new school datasets will be created.
outputDir = 'C:/Users/Jake/Documents/GitHub/EducationDataNC/' + str(schoolYear) + '/Machine Learning Datasets/'

#Missing Data Threshold (Per Column)
missingThreshold = 0.60

#Unique Value Threshold (Per Column)
#Delete Columns >  uniqueThreshold unique values prior to one-hot encoding. 
#(each unique value becomes a new column during one-hot encoding)
uniqueThreshold = 25

#Read in the School Data File
schoolData = pd.read_csv(inputFile, low_memory=False, dtype={'unit_code': object})
print('*********Start: Beginning Column and Row Counts********************************************')
schoolData.info(verbose=False)

#Select only public schools as charter schools are missing data for many columns.
schoolData = schoolData[(schoolData['type_cd'] == 'P') & (schoolData['student_num'] > 0)]

print('\r\n*********After: Selecting Only Public School Campuses**********************************')
schoolData.info(verbose=False)

#Save primary key
unit_code = schoolData['unit_code']
#Convert zip code to string
schoolData['szip_ad'] = schoolData['szip_ad'].astype('object')

#Get Student Body Racial Composition Fields
raceCompositionFields = schoolData.filter(regex='Indian|Asian|Hispanic|Black|White|PacificIsland|TwoOrMore|Minority')                                  .filter(regex='Pct').columns
    
rowsBefore = schoolData[raceCompositionFields].isnull().T.any().T.sum()

#Update missing race values with the district average when avaiable (No district averages for charter schools) 
schoolData[raceCompositionFields] = schoolData.groupby('District Name')[raceCompositionFields]                                              .transform(lambda x: x.fillna(x.mean()))

    #Review dataset contents after Racial Composition Imputation
print('*********After: Updating Missing Racial Compostion Values****************************')   
rowsAfter = schoolData[raceCompositionFields].isnull().T.any().T.sum()
rowsUpdated = rowsBefore - rowsAfter
print 'Rows Updated / Imputed: ', rowsUpdated 
print('\r\nTotal Rows Missing Racial Compositions By District Name') 
schoolData['District Name'][schoolData[raceCompositionFields].isnull().T.any().T].value_counts()

#Remove any fields that have the same value in all rows
UniqueValueCounts = schoolData.nunique(dropna=False)
SingleValueCols = UniqueValueCounts[UniqueValueCounts == 1].index
schoolData = schoolData.drop(SingleValueCols, axis=1)

#Review dataset contents after drops
print('*********After: Removing columns with the same value in every row.*******************')
schoolData.info(verbose=False)
print '\r\nColumns Deleted: ', len(SingleValueCols)

#Remove any fields that have unique values in every row
schoolDataRecordCt = schoolData.shape[0]
UniqueValueCounts = schoolData.apply(pd.Series.nunique)
AllUniqueValueCols = UniqueValueCounts[UniqueValueCounts == schoolDataRecordCt].index
schoolData = schoolData.drop(AllUniqueValueCols, axis=1)

#Review dataset contents after drops
print('*********After: Removing columns with unique values in every row.*******************')
schoolData.info(verbose=False)
print '\r\nColumns Deleted: ', len(AllUniqueValueCols)

#Remove any empty fields (null values in every row)
schoolDataRecordCt = schoolData.shape[0]
NullValueCounts = schoolData.isnull().sum()
NullValueCols = NullValueCounts[NullValueCounts == schoolDataRecordCt].index
schoolData = schoolData.drop(NullValueCols, axis=1)

#Review dataset contents after empty field drops
print('*********After: Removing columns with null / blank values in every row.*************')
schoolData.info(verbose=False)
print '\r\nColumns Deleted: ', len(NullValueCols)

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
schoolDataRecordCt = sD_continuous.shape[0]
missingValueLimit = schoolDataRecordCt * missingThreshold
NullValueCounts = sD_continuous.isnull().sum()
NullValueCols = NullValueCounts[NullValueCounts >= missingValueLimit].index
schoolData = schoolData.drop(NullValueCols, axis=1)

#Review dataset contents after empty field drops
print('*********After: Removing columns with >= missingThreshold % of missing values******')
schoolData.info(verbose=False)
print '\r\nColumns Deleted: ', len(NullValueCols)

#Delete categorical columns with > 25 unique values (Each unique value becomes a column during one-hot encoding)
oneHotUniqueValueCounts = schoolData[sD_nominal.columns].apply(lambda x: x.nunique())
oneHotUniqueValueCols = oneHotUniqueValueCounts[oneHotUniqueValueCounts >= uniqueThreshold].index
schoolData.drop(oneHotUniqueValueCols, axis=1, inplace=True) 

#Review dataset contents one hot high unique value drops
print('*********After: Removing columns with >= uniqueThreshold unique values***********')
schoolData.info(verbose=False)
print '\r\nColumns Deleted: ', len(oneHotUniqueValueCols)

#Isolate remaining categorical variables
begColumnCt = len(schoolData.columns)
sD_nominal = schoolData.loc[:, (schoolData.dtypes == object)]

#one hot encode categorical variables
schoolData = pd.get_dummies(data=schoolData, 
                       columns=sD_nominal, drop_first=True)

#Determine change in column count
endColumnCt = len(schoolData.columns)
columnsAdded = endColumnCt - begColumnCt

#Review dataset contents one hot high unique value drops
print 'Columns To One-Hot Encode: ', len(sD_nominal.columns)
print('\r\n*********After: Adding New Columns Via One-Hot Encoding*************************')
schoolData.info(verbose=False)
print '\r\nNew Columns Created Via One-Hot Encoding: ', columnsAdded

#Print out all the missing value rows
pd.set_option('display.max_rows', 1000)

print('\r\n*********The Remaining Missing Values Below will be set to Zero!*************************')

#Check for Missing values 
missing_values = schoolData.isnull().sum().reset_index()
missing_values.columns = ['Variable Name', 'Number Missing Values']
missing_values = missing_values[missing_values['Number Missing Values'] > 0] 
missing_values


#Replace all remaining NaN with 0
schoolData = schoolData.fillna(0)

#Check for Missing values after final imputation 
missing_values = schoolData.isnull().sum().reset_index()
missing_values.columns = ['Variable Name', 'Number Missing Values']
missing_values = missing_values[missing_values['Number Missing Values'] > 0] 
missing_values

# calculate the correlation matrix
corr_matrix  = schoolData.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

#Get all of the correlation values > 95%
x = np.where(upper > 0.95)

#Display all field combinations with > 95% correlation
cf = pd.DataFrame()
cf['Field1'] = upper.columns[x[1]]
cf['Field2'] = upper.index[x[0]]

#Get the correlation values for every field combination. (There must be a more pythonic way to do this!)
corr = [0] * len(cf)
for i in range(0, len(cf)):
    corr[i] =  upper[cf['Field1'][i]][cf['Field2'][i]] 
    
cf['Correlation'] = corr

print 'There are ', str(len(cf['Field1'])), ' field correlations > 95%.'
cf

print 'Dropping the following ', str(len(to_drop)), ' highly correlated fields.'
to_drop

#Check columns before drop 
print('\r\n*********Before: Dropping Highly Correlated Fields*************************************')
schoolData.info(verbose=False)

# Drop the highly correlated features from our training data 
schoolData = schoolData.drop(to_drop, axis=1)

#Check columns after drop 
print('\r\n*********After: Dropping Highly Correlated Fields**************************************')
schoolData.info(verbose=False)

#Restore the unit_code before saving
schoolData['unit_code'] = unit_code
#Save the final dataset to a .csv file
schoolData.to_csv(outputDir + inputFileName + '_ML.csv', sep=',', index=False)

print('*********FINAL DATASET DETAILS*********************************************************\r\n')
schoolData.info(verbose=True)

import sklearn
import pandas as pd

print('Sklearn Version: ' + sklearn.__version__)
print('Pandas Version: ' + pd.__version__)

print 'Output File Location:\r\n\r\n' + outputDir + inputFileName + '_ML.csv'



