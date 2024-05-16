import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
get_ipython().run_line_magic('matplotlib', 'inline')

#Read the dataset in and get a quick look at the underlying data
mass_shootings = pd.read_csv('Mass Shootings Dataset Ver 5.csv', encoding = "ISO-8859-1", parse_dates=['Date'])
mass_shootings.head(5)

#Get year in its own column, create a variable that counts totals
mass_shootings['Year'] = mass_shootings['Date'].dt.year
count_by_year = mass_shootings['Year'].value_counts()
fatalities = mass_shootings[['Year','Fatalities']].groupby('Year').sum()

#Now set plot preferences and show plot
plt.figure(figsize=(18, 7))
plt.plot(count_by_year.sort_index(), color = 'r', linewidth = 3, label='Shootings per Year')
plt.plot(fatalities.sort_index(), color = 'b', linewidth = 3, label='Fatalities per Year')
plt.xticks(count_by_year.index, rotation = 'vertical', fontsize=12)
plt.xlabel('Mass Shooting Year', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Number of Mass Shootings and Fatalities Per Year', fontsize=30)
plt.legend(fontsize=15)
plt.show()

#Print out .describe() method for both series
print(count_by_year.describe())
print(fatalities.describe())

#Create variable to show fatalities, injuries, total victims grouped by year
year_in_depth = mass_shootings[['Year','Fatalities','Injured','Total victims']].groupby('Year').sum()

#Set plot preferences and show plot
year_in_depth.plot.bar(figsize=(30, 12))
plt.legend(fontsize=20)
plt.title('Shootings by Year - Fatalities, Injuries, and Total Victims', fontsize=33)
plt.xlabel('Mass Shooting Year', fontsize=25)
plt.ylabel('Count of Shootings', fontsize=25)
plt.xticks(fontsize=18)

#Describe the data
year_in_depth.describe()

#Let's see if there was one large incident that led to the skewed count. Let's pick
#150 as a starting point.

mass_shootings[mass_shootings['Total victims'] > 150]['Total victims']

#Separate our 'Location' column which currently gives us
#a "City, State" format into two new columns for 'City' and 'State'

for i in mass_shootings['Location']:
    mass_shootings['City'] = mass_shootings['Location'].str.partition(',')[0]
    mass_shootings['State'] = mass_shootings['Location'].str.partition(',')[2]

#Now print the head of our new 'State' column
mass_shootings['State'].head(10)

#Decided to convert the format to state abbreviations. Due to some data 
#having more than one city listed in the 'Location' column and therefore more than 
#one ',' which is what our partition method was based on, we need to re-categorize some
#of this data.

mass_shootings['State'].replace([' TX', ' CO', ' MD', ' NV', ' CA', ' PA', ' Florida', ' Ohio',
       ' California', ' WA', ' LA', ' Texas', ' Missouri',
       ' Virginia', ' North Carolina', ' Tennessee', ' Texas ',
       ' Kentucky', ' Alabama', ' Pennsylvania', ' Kansas',
       ' Massachusetts', '  Virginia', ' Washington', ' Arizona',
       ' Michigan', ' Mississippi', ' Nebraska', ' Colorado',
       ' Minnesota', ' Georgia', ' Maine', ' Oregon', ' South Dakota',
       ' New York', ' Louisiana', ' Illinois', ' South Carolina',
       ' Wisconsin', ' Montana', ' New Jersey', ' Indiana', ' Oklahoma',
       ' New Mexico', ' Idaho',
       ' Souderton, Lansdale, Harleysville, Pennsylvania',
       ' West Virginia', ' Nevada', ' Albuquerque, New Mexico',
       ' Connecticut', ' Arkansas', ' Utah', ' Lancaster, Pennsylvania',
       ' Vermont', ' San Diego, California', ' Hawaii', ' Alaska',
       ' Wyoming', ' Iowa'], ['TX', 'CO', 'MD', 'NV', 'CA', 'PA', 'FL', 'OH', 'CA', 'WA', 'LA',
        'TX', 'MO', 'VA', 'NC', 'TN', 'TX', 'KY', 'AL', 'PA', 'KS', 'MA', 'VA', 'WA', 'AZ', 'MI',
        'MS', 'NE', 'CO', 'MN', 'GA', 'ME', 'OR', 'SD', 'NY', 'LA', 'IL', 'SC', 'WI', 'MT',
        'NJ', 'IN', 'OK', 'NM', 'ID', 'PA', 'WV', 'NV', 'NM', 'CT', 'AR', 'UT',
        'PA', 'VT', 'CA', 'HI', 'AL', 'WY', 'IA'], inplace=True)

#Create dataframe without NA's
mass_shootings_state_without_na = pd.DataFrame(mass_shootings['State'].dropna())

#Confirm that this worked
mass_shootings_state_without_na['State'].unique()

#Finally, now that our state data is clean, we can take a look at the number of shootings by
#state.

#Create variable looking just at state value counts
shooting_by_state = mass_shootings_state_without_na['State'].value_counts()

#Now set plot preferences and show plot
plt.figure(figsize=(15, 5))
plt.bar(shooting_by_state.index, shooting_by_state.values)
plt.xticks(rotation = 'vertical', fontsize=12)
plt.xlabel('State Abbreviation', fontsize=15)
plt.ylabel('Count of Shootings', fontsize=15)
plt.title('Number of Shootings by State', fontsize=25)
plt.show()

#Describe the data
shooting_by_state.describe()

#Create variable for only our top states, then another variable which groups them by fatalities
#injuries, and total victims
highest_states = mass_shootings[mass_shootings["State"].isin(['CA', 'FL', 'TX'])]
deadliest_state = highest_states[['State','Fatalities','Injured', 'Total victims']].groupby('State').sum()

#Set plot preferences and show plot
deadliest_state.plot.bar(figsize=(18, 8))
plt.legend(fontsize=15)
plt.title('Shootings by State - Fatalities, Injuries, and Total Victims', fontsize=25)
plt.xlabel('State', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.xticks(fontsize=15, rotation='horizontal')

#First, create our 'Fatality Rate' measure by creating a new column. Next create a variable for
#just that new column from our mass_shootings dataframe.
mass_shootings['Fatality Rate'] = mass_shootings['Fatalities']/mass_shootings['Total victims']
fatality_rate = mass_shootings['Fatality Rate']

#Set plot preferences
plt.figure(figsize=(10, 7))
plt.hist(mass_shootings['Fatality Rate'], bins=4, color='b')
plt.xlabel('Fatality Rate', fontsize=15)
plt.ylabel('Occurrences of Fatality Rate', fontsize=15)
plt.title('Distribution of Fatality Rate', fontsize=25)

#Plot the distribution and show mean/standard deviation to get a feel for whether or not those
#descriptive statistics are a good measure. Then show the plot.
plt.axvline(fatality_rate.mean(), color='r', linestyle='solid', linewidth=3)
plt.axvline(fatality_rate.mean() + fatality_rate.std(), color='r', linestyle='dashed', linewidth=3)
plt.axvline(fatality_rate.mean()-fatality_rate.std(), color='r', linestyle='dashed', linewidth=3) 

plt.show()

#Describe the data
print(fatality_rate.mean())
print(fatality_rate.max())

pd.set_option('display.max_columns', 25)
mass_shootings[mass_shootings["Fatalities"] > mass_shootings['Total victims']].head(5)

#Let's first look at location at an aggregate level

#Need to recategorize based on typo in our data (Open+CLose). Identify the row and change
#that value to be consistent with 'Open+Close'. Find the index for incorrect value:
mass_shootings.loc[mass_shootings['Open/Close Location'] == 'Open+CLose']

#Now, change the value based on index 280
mass_shootings.at[280, 'Open/Close Location'] = 'Open+Close'

#Create variable to show total victims, fatalities, and injuries gropued by Open/Close Location
open_close = mass_shootings[['Open/Close Location', 'Total victims', 'Fatalities', 'Injured']].groupby('Open/Close Location').sum()

#Set plot preferences and show plot
open_close.plot.bar(figsize=(15,5))
plt.title('Shootings by Location - Total Victims, Fatalities, and Injuries', fontsize=20)
plt.xlabel('Shooting Location', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.legend(fontsize=15)
plt.xticks(rotation='horizontal', fontsize=12)

#Create two dataframes looking solely at shooting locations of 'Open' and 'Close'
open_location = pd.DataFrame(mass_shootings[mass_shootings['Open/Close Location'] == 'Open'])
closed_location = pd.DataFrame(mass_shootings[mass_shootings['Open/Close Location'] == 'Close'])

#Based on what we saw earlier, only take fatality rates less than or equal to 1
open_location = open_location[open_location['Fatality Rate'] <= 1]
closed_location = closed_location[closed_location['Fatality Rate'] <= 1]

#Set plot preferences. Want to plot 2 histograms (in subplot fashion) of distribution of 
#fatality rate based on shooting location

plt.figure(figsize=[20,5])

plt.subplot(1, 3, 1)
plt.hist(open_location['Fatality Rate'], bins=4, color='r', alpha=.5, label='Open')
plt.axvline(open_location['Fatality Rate'].mean(), color='b', linestyle='solid', linewidth=3)
plt.title('Fatality Rate Distribution in Open Shooting Locations')
plt.ylabel('Count', fontsize=13)
plt.xlabel('Fatality Rate', fontsize=13)

plt.subplot(1, 3, 2)
plt.hist(closed_location['Fatality Rate'], bins=4, color='b', alpha=.5, label='Closed')
plt.axvline(closed_location['Fatality Rate'].mean(), color='r', linestyle='solid', linewidth=3)
plt.title('Fatality Rate Distribution in Closed Shooting Locations')
plt.ylabel('Count', fontsize=13)
plt.xlabel('Fatality Rate', fontsize=13)

plt.show()

#Calculate T Statistic and P Value

# Difference in means
diff=open_location['Fatality Rate'].mean( ) -closed_location['Fatality Rate'].mean()

#Set variables for sample size and standard deviations
size = np.array([len(open_location['Fatality Rate']), len(closed_location['Fatality Rate'])])
sd = np.array([open_location['Fatality Rate'].std(), closed_location['Fatality Rate'].std()])

# The squared standard deviations are divided by the sample size and summed, then we take
# the square root of the sum. 
diff_se = (sum(sd ** 2 / size)) ** 0.5  

#Print p value and t statistic
from scipy.stats import ttest_ind
print(ttest_ind(closed_location['Fatality Rate'], open_location['Fatality Rate'], equal_var=False))

#Let's start by looking at the distribution of shooter age. There are a couple instances where
#more than one shooter is present, let's try to separate those out first. 

def split_age_second_shooter(age):
   second_shooter_age = age.split(',')
   if len(second_shooter_age) == 2:
       return second_shooter_age[1]
   else:
       return 0

def split_age_first_shooter(age):
   first_shooter_age = age.split(',')
   if len(first_shooter_age) == 2:
       return first_shooter_age[0]
   else:
       return age

#Create new columns for 'First Shooter Age' and 'Second Shooter Age' and then apply our functions above

mass_shootings['Age'] = mass_shootings['Age'].astype(str)
mass_shootings['Second Shooter Age'] = mass_shootings['Age'].apply(split_age_second_shooter)
mass_shootings['First Shooter Age'] = mass_shootings['Age'].apply(split_age_first_shooter)

#Set plot preferences and show plot. Let's look specifically at the primary shooter.
plt.figure(figsize=(17, 8))
plt.scatter(x=mass_shootings['First Shooter Age'], y=mass_shootings['Fatalities'])
plt.xlabel('Shooter Age', fontsize=20)
plt.xticks(rotation=45, fontsize=12)
plt.ylabel('Fatalities', fontsize=20)
plt.title('Shooter Age vs. Number of Fatalities', fontsize=25)
plt.legend(fontsize=15, loc='best')

plt.show()

#Next, let's see if we can dissect the 'Cause' column. After running a unique() method I
#found that several causes could be combined to provide us with some more concrete results.
#For example 'anger' should be combined into one category absorbing 'frustration' and 'revenge'.

mass_shootings['Cause'].replace(['unknown', 'terrorism', 'unemployement', 'racism',
       'frustration', 'domestic dispute', 'anger', 'psycho', 'revenge',
       'domestic disputer', 'suspension', 'religious radicalism', 'drunk',
       'failing exams', 'breakup', 'robbery'], ['Unknown', 'Terrorism', 'Unemployment', 'Racism',
        'Anger', 'Domestic Dispute', 'Anger', 'Pyschotic', 'Anger', 'Domestic Dispute',
        'Suspension', 'Religious Radicalism', 'Drunk', 'Failing Exams', 'Breakup', 
        'Robbery'], inplace=True)

#Create dataframe without NA's
mass_shootings_cause_without_na = pd.DataFrame(mass_shootings['Cause'].dropna())

#Confirm that this worked
mass_shootings_cause_without_na['Cause'].unique()

#Create variable for cause value counts
cause = mass_shootings_cause_without_na['Cause'].value_counts()

#Set plot preferences and show plot
plt.figure(figsize=(15, 5))
plt.bar(cause.index, cause.values)
plt.xlabel('Cause', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Number of Shootings by Shooter Cause', fontsize=20)
plt.xticks(rotation = 45)

plt.show()

#Next, let's look at race and gender. We want to see fatalities, injuries and total victims by 
#race first. Similar to above, better groupings must be established.

mass_shootings['Race'].replace(['White', 'Black', 'Asian', 'Latino', 'Other', 'Unknown',
       'Black American or African American',
       'White American or European American', 'Asian American',
       'Some other race', 'Two or more races',
       'Black American or African American/Unknown',
       'White American or European American/Some other Race',
       'Native American or Alaska Native', 'white', 'black',
       'Asian American/Some other race'], ['White', 'Black', 'Asian', 'Latino', 'Other', 
        'Other', 'Black', 'White', 'Asian', 'Other', 'Other', 'Black', 'White', 
        'Native', 'White', 'Black', 'Asian'], inplace=True)

#Now that race categories are established, set plot preferences and show plot
mass_shootings[['Race', 'Fatalities']].boxplot(by='Race')
plt.suptitle('')
plt.xticks(rotation='vertical')
plt.title('Fatalities by Race', fontsize=18)
plt.xticks(rotation=45, )
plt.xlabel('Race', fontsize=15)

#Now, let's look at gender. First, we need to recreate the groupings
mass_shootings['Gender'].replace(['M', 'Unknown', 'Male', 'M/F', 'Male/Female', 'Female'],
['Male', 'Unknown', 'Male', 'Male & Female', 'Male & Female',
'Female'], inplace = True)

#Now that gender categories are established, set plot preferences and show plot
mass_shootings[['Gender', 'Fatalities']].boxplot(by='Gender')
plt.suptitle('')
plt.xticks(rotation='vertical')
plt.title('Fatalities by Gender', fontsize=18)
plt.xticks(rotation=45, )
plt.xlabel('Gender', fontsize=15)

#Now let's look at mental health. I am interested to know the number of individuals with 
#known mental health issues. Again, need to combine categories first.

mass_shootings['Mental Health Issues'].replace(['No', 'Unclear', 'Yes', 'Unknown', 'unknown'],
['No', 'Unknown', 'Yes', 'Unknown', 'Unknown'], inplace=True)

#Now that mental health categories are established, create variable to show value counts for all
#categories
mental_health = mass_shootings['Mental Health Issues'].value_counts()

#Set plot preferences and show plot
mental_health.plot.bar(figsize=(8, 5))
plt.title('Breakdown of Shooters with Mental Health Issues', fontsize=15)
plt.xticks(rotation='horizontal')
plt.xlabel('Count', fontsize=13)
plt.ylabel('Known Mental Health Issues', fontsize=13)

