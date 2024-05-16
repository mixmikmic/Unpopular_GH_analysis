import numpy as np
import pandas as pd
import scipy as scipy
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import cprint
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv("/home/gierkep/google-drive/Datasets/Student Alcohol Consump/student-mat.csv", low_memory = False) 

df.head()

df.shape #prints the shape of the dataframe

df.describe()

df.isnull().values.any()  #check for missing data

def checkp (type1, type2): 
    '''
    We used the Mann-Whitney test here for the following reasons:
    Mann-Whitney is a more robust test because it is less likely than the T-test to indicate significance due to
    the presence of outliers.  The Mann-Whitney test has a 0.95 efficiency when compared to the t-test and for 
    non-normalized distributions, which our datasets could be, or larger distributions the Mann-Whitney test is
    considerably more efficient.
    '''

    ptest = scipy.stats.mannwhitneyu(type1['G3'], type2['G3']).pvalue 
    print('Mann-Whitney', 'G3', scipy.stats.mannwhitneyu(type1['G3'], type2['G3']))
    if ptest > 0.05:
        cprint("The pvalue is greater than 0.05 indicating the null hypothesis", 'red') #prints evaluation of pvalue
    else:
        cprint("The pvalue is less than 0.05 so we reject the null hypothesis", 'green') #prints evaluation of pvalue

def histvsdata(set1,set1title, set2,set2title, plottitle,plotdata):
    plt.figure(figsize=(5,10))
    plt.subplot(2, 1, 1)
    plt.hist(set1[plotdata], label = set1title, alpha = 0.5, normed = True)
    plt.axvline(set1[plotdata].mean(), color='b', linestyle='solid', linewidth=2)
    plt.title(plottitle)
    plt.hist(set2[plotdata], label = set2title, alpha = 0.5, normed = True)
    plt.axvline(set2[plotdata].mean(), color='g', linestyle='solid', linewidth=2)
    plt.legend(loc='upper left')
    plt.show()

corrmat = df.corr(method='spearman') #use spearman method because all the data is ordinal
f, ax = plt.subplots(figsize=(10, 10))
# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True,annot=True,fmt='.2f')
plt.show()

df['Dalc'].unique() #Dalc us a 1-5 ranked scale of workday drinking habits

df['Dalc'].mean()

drinksaboveavg = df[df['Dalc']>=df['Dalc'].mean()] #create dataframe of the above average workday drinkers
drinksaboveavg.name = 'drinksaboveavg'
drinksbelowavg = df[df['Dalc']<df['Dalc'].mean()] #create dataframe of the below average workday drinkers
drinksbelowavg.name = 'drinksbelowavg'
checkp(drinksbelowavg,drinksaboveavg)

histvsdata(drinksaboveavg, 'Drinks above average', 
         drinksbelowavg, 'Drinks less than average', 
         'Weekday Drinking vs G3 scores',
        'G3')

df['Walc'].unique()  #Walc us a 1-5 ranked scale of weekend drinking habits

df['Walc'].mean()

drinksaboveavgwd = df[df['Walc']>=df['Walc'].mean()] #create dataframe of the above average weekend drinkers
drinksaboveavgwd.name = 'drinksaboveavgweekend'
drinksbelowavgwd = df[df['Walc']<df['Walc'].mean()] #create dataframe of the below average weekend drinkers
drinksbelowavgwd.name = 'drinksbelowavgweekend'
checkp(drinksbelowavg,drinksaboveavg)

histvsdata(drinksaboveavgwd, 'Drinks above average weekends', 
         drinksbelowavgwd, 'Drinks less than average weekends', 
         'Weekend Drinking vs G3 scores',
        'G3')

df['Mjob'].unique()

stayathomemom = df[df['Mjob']=='at_home'] #create a dataframe of students where the mother is a stay at home mom
stayathomemom.name = 'stayathomemom'
notstayathomemom = df[df['Mjob']!='at_home'] #create a dataframe of students where the mother is a working mom
notstayathomemom.name = 'notstayathomemom'

checkp(stayathomemom,notstayathomemom)    #checks for correlations of pvalues that at less than 0.01 suggesting intesting data

histvsdata(stayathomemom, 'Stay at home Mother', 
         notstayathomemom, 'Working Mother', 
         'Mother work status vs G3 scores',
          'G3')

from scipy.stats import mstats

Grp_1 = df[df['Mjob']=='at_home']['G3'] #Filter student performance for mothers with no job
Grp_2 = df[df['Mjob']=='health']['G3'] #Filter student performance for mothers working in healthcare
Grp_3 = df[df['Mjob']=='other']['G3'] #Filter student performance for mothers  working in other
Grp_4 = df[df['Mjob']=='services']['G3'] #Filter student performance for mothers working in services
Grp_5 = df[df['Mjob']=='teacher']['G3'] #Filter student performance for mothers working in education

print("Kruskal Wallis H-test test:")

#Since we are dealing with more than two datasets well use Kruskal-Willis to test for significance
H, pval = mstats.kruskalwallis(Grp_1, Grp_2, Grp_3, Grp_4, Grp_5)

print("H-statistic:", H)
print("P-Value:", pval)

if pval < 0.05:
    cprint("Reject NULL hypothesis - Significant differences exist between groups.",'green')
if pval > 0.05:
    cprint("Accept NULL hypothesis - No significant difference between groups.", 'red')

plt.figure(figsize=(5,10))

plt.subplot(2, 1, 1)
plt.title('Mothers Employment against G3')
plt.xlim(0, 20)
plt.hist(Grp_1, label = 'at_home', normed = True, fill=False, histtype='step', linewidth = 2)
plt.hist(Grp_2, label = 'health', normed = True, fill= False, histtype='step', linewidth = 2)
plt.hist(Grp_3, label = 'other', normed = True, fill= False, histtype='step', linewidth = 2)
plt.legend(loc='upper left')

plt.subplot(2, 1, 2)    
plt.xlim(0, 20)
plt.hist(Grp_4, label = 'services', normed = True, fill= False, histtype='step', linewidth = 2)
plt.hist(Grp_5, label = 'teacher', normed = True, fill= False, histtype='step', linewidth = 2)
plt.legend(loc='upper left')

plt.show()

df['activities'].unique()

aftrschlactive = df[df['activities']=='yes']  #create a dataframe where students have after school activities
aftrschlactive.name = 'aftrschlactive'
noaftrschlactive = df[df['activities']=='no'] #create a dataframe where students dont have after school activities
noaftrschlactive.name = 'noaftrschlactive'

checkp(aftrschlactive,noaftrschlactive)  #checks for correlations of pvalues that at less than 0.01 suggesting intesting data

histvsdata(aftrschlactive, 'Has after school activities', 
         noaftrschlactive, 'No after school activities', 
         'Has after school activities vs G3 scores',
        'G3')

df['romantic'].unique()

romantic = df[df['romantic']=='yes']  #create a dataframe where students have romantic relationships
romantic.name = 'romantic'
noromantic = df[df['romantic']=='no'] #create a dataframe where students dont have romantic relationships
noromantic.name = 'noromantic'

checkp(romantic,noromantic)  #checks for correlations of pvalues that at less than 0.01 suggesting intesting data

histvsdata(romantic, 'In a relationship', 
         noromantic, 'Not in a relationship', 
         'Students relationships status vs G3 scores',
        'G3')

df['higher'].unique()

wantshigher = df[df['higher']=='yes']  #create a dataframe where students want to get higher education
wantshigher.name = 'wantshigher'
nohigherdesire = df[df['higher']=='no'] #create a dataframe where students dont want to get higher education
nohigherdesire.name = 'nohigherdesire'

checkp(wantshigher,nohigherdesire)  #checks for correlations of pvalues that at less than 0.01 suggesting intesting data

histvsdata(wantshigher, 'Wants higher education', 
         nohigherdesire, 'Doesnt want Higher Edu', 
         'Desire to continue studies vs G3 scores',
        'G3')



