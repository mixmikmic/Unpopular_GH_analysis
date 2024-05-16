import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import gender_guesser.detector as gender
import time
import collections
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=1.5)

def run_query(query):
    with sqlite3.connect('database.sqlite') as conn:
        return pd.read_sql(query, conn)

#Read the data from SQL->Pandas
q1 = '''
SELECT * FROM Salaries
'''

data = run_query(q1)
data.head()

data.dtypes

data['JobTitle'].nunique()

data['Year'].value_counts()

data['Notes'].value_counts()

data['Agency'].value_counts()

data['Status'].value_counts()

def process_pay(df):
    cols = ['BasePay','OvertimePay', 'OtherPay', 'Benefits']
    
    print('Checking for nulls:')
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors ='coerce')
        print(len(col)*'-')
        print(col)
        print(len(col)*'-')
        print(df[col].isnull().value_counts())
        
    return df

data = process_pay(data.copy())

def process_pay2(df):
    df['Benefits'] = df['Benefits'].fillna(0)
    
    df = df.dropna()
    print(df['BasePay'].isnull().value_counts())
    return df

data = process_pay2(data)

data = data.drop(columns=['Agency', 'Notes'])

#Create the 'Gender' column based on employee's first name.
d = gender.Detector(case_sensitive=False)
data['FirstName'] = data['EmployeeName'].str.split().apply(lambda x: x[0])
data['Gender'] = data['FirstName'].apply(lambda x: d.get_gender(x))
data['Gender'].value_counts()

#Retain data with 'male' and 'female' names.
male_female_only = data[(data['Gender'] == 'male') | (data['Gender'] == 'female')].copy()
male_female_only['Gender'].value_counts()

def find_job_title2(row):
    
    #Prioritize specific titles on top 
    titles = collections.OrderedDict([
        ('Police',['police', 'sherif', 'probation', 'sergeant', 'officer', 'lieutenant']),
        ('Fire', ['fire']),
        ('Transit',['mta', 'transit']),
        ('Medical',['anesth', 'medical', 'nurs', 'health', 'physician', 'orthopedic', 'pharm', 'care']),
        ('Architect', ['architect']),
        ('Court',['court', 'legal']),
        ('Mayor Office', ['mayoral']),
        ('Library', ['librar']),
        ('Public Works', ['public']),
        ('Attorney', ['attorney']),
        ('Custodian', ['custodian']),
        ('Gardener', ['garden']),
        ('Recreation Leader', ['recreation']),
        ('Automotive',['automotive', 'mechanic', 'truck']),
        ('Engineer',['engineer', 'engr', 'eng', 'program']),
        ('General Laborer',['general laborer', 'painter', 'inspector', 'carpenter', 'electrician', 'plumber', 'maintenance']),
        ('Food Services', ['food serv']),
        ('Clerk', ['clerk']),
        ('Porter', ['porter']),
        ('Airport Staff', ['airport']),
        ('Social Worker',['worker']),        
        ('Guard', ['guard']),
        ('Assistant',['aide', 'assistant', 'secretary', 'attendant']),        
        ('Analyst', ['analy']),
        ('Manager', ['manager'])      
    ])       
         
    #Loops through the dictionaries
    for group, keywords in titles.items():
        for keyword in keywords:
            if keyword in row.lower():
                return group
    return 'Other'

start_time = time.time()    
male_female_only["Job_Group"] = male_female_only["JobTitle"].map(find_job_title2)
print("--- Run Time: %s seconds ---" % (time.time() - start_time))

male_female_only['Job_Group'].value_counts()

fig = plt.figure(figsize=(10, 5))
male_only = male_female_only[male_female_only['Gender'] == 'male']
female_only = male_female_only[male_female_only['Gender'] == 'female']


ax = sns.kdeplot(male_only['TotalPayBenefits'], color ='Blue', label='Male', shade=True)
ax = sns.kdeplot(female_only['TotalPayBenefits'], color='Red', label='Female', shade=True)

plt.yticks([])
plt.title('Overall Income Distribution')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay + Benefits ($)')
plt.xlim(0, 350000)
plt.show()

fig = plt.figure(figsize=(5, 5))

colors = ['#AFAFF5', '#EFAFB5']
labels = ['Male', 'Female']
sizes = [len(male_only), len(female_only)]
explode = (0.05, 0)
sns.set(font_scale=1.5)
ax = plt.pie(sizes, labels=labels, explode=explode, colors=colors, shadow=True, startangle=90, autopct='%1.f%%')

plt.title('Estimated Percentages of Employees: Overall')
plt.show()

data_2011 = male_female_only[male_female_only['Year'] == 2011]
data_2012 = male_female_only[male_female_only['Year'] == 2012]
data_2013 = male_female_only[male_female_only['Year'] == 2013]
data_2014 = male_female_only[male_female_only['Year'] == 2014]


plt.figure(figsize=(10,7.5))
ax = plt.boxplot([data_2011['TotalPayBenefits'].values, data_2012['TotalPayBenefits'].values,                   data_2013['TotalPayBenefits'].values, data_2014['TotalPayBenefits'].values])
plt.ylim(0, 350000)
plt.xticks([1, 2, 3, 4], ['2011', '2012', '2013', '2014'])
plt.xlabel('Year')
plt.ylabel('Total Pay + Benefits ($)')
plt.tight_layout()

years = ['2011', '2012', '2013', '2014']
all_data = [data_2011, data_2012, data_2013, data_2014]

for i in range(4):
    print(len(years[i])*'-')
    print(years[i])
    print(len(years[i])*'-')
    print(all_data[i]['Status'].value_counts())

data_2014_FT = data_2014[data_2014['Status'] == 'FT']
data_2014_PT = data_2014[data_2014['Status'] == 'PT']

fig = plt.figure(figsize=(10, 5))
ax = sns.kdeplot(data_2014_PT['TotalPayBenefits'], color = 'Orange', label='Part Time Workers', shade=True)
ax = sns.kdeplot(data_2014_FT['TotalPayBenefits'], color = 'Green', label='Full Time Workers', shade=True)
plt.yticks([])

plt.title('Part Time Workers vs. Full Time Workers')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay + Benefits ($)')
plt.xlim(0, 350000)
plt.show()

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(hspace=.5)  

#Generate the top plot
male_only = data_2014_FT[data_2014_FT['Gender'] == 'male']
female_only = data_2014_FT[data_2014_FT['Gender'] == 'female']
ax = fig.add_subplot(2, 1, 1)
ax = sns.kdeplot(male_only['TotalPayBenefits'], color ='Blue', label='Male', shade=True)
ax = sns.kdeplot(female_only['TotalPayBenefits'], color='Red', label='Female', shade=True)
plt.title('San Francisco: Full Time Workers')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay & Benefits ($)')
plt.xlim(0, 400000)
plt.yticks([])

#Generate the bottom plot
male_only = data_2014_PT[data_2014_PT['Gender'] == 'male']
female_only = data_2014_PT[data_2014_PT['Gender'] == 'female']
ax2 = fig.add_subplot(2, 1, 2)
ax2 = sns.kdeplot(male_only['TotalPayBenefits'], color ='Blue', label='Male', shade=True)
ax2 = sns.kdeplot(female_only['TotalPayBenefits'], color='Red', label='Female', shade=True)
plt.title('San Francisco: Part Time Workers')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay & Benefits ($)')
plt.xlim(0, 400000)
plt.yticks([])

plt.show()

male_only = data_2014_FT[data_2014_FT['Gender'] == 'male']
female_only = data_2014_FT[data_2014_FT['Gender'] == 'female']

fig = plt.figure(figsize=(10, 15))
fig.subplots_adjust(hspace=.5)  

#Generate the top plot  
ax = fig.add_subplot(3, 1, 1)
ax = sns.kdeplot(male_only['OvertimePay'], color ='Blue', label='Male', shade=True)
ax = sns.kdeplot(female_only['OvertimePay'], color='Red', label='Female', shade=True)
plt.title('Full Time Workers')
plt.ylabel('Density of Employees')
plt.xlabel('Overtime Pay ($)')
plt.xlim(0, 60000)
plt.yticks([])

#Generate the middle plot
ax2 = fig.add_subplot(3, 1, 2)
ax2 = sns.kdeplot(male_only['Benefits'], color ='Blue', label='Male', shade=True)
ax2 = sns.kdeplot(female_only['Benefits'], color='Red', label='Female', shade=True)
plt.ylabel('Density of Employees')
plt.xlabel('Benefits Only ($)')
plt.xlim(0, 75000)
plt.yticks([])

#Generate the bottom plot
ax3 = fig.add_subplot(3, 1, 3)
ax3 = sns.kdeplot(male_only['BasePay'], color ='Blue', label='Male', shade=True)
ax3 = sns.kdeplot(female_only['BasePay'], color='Red', label='Female', shade=True)
plt.ylabel('Density of Employees')
plt.xlabel('Base Pay Only  ($)')
plt.xlim(0, 300000)
plt.yticks([])

plt.show()

data_2014_FT.corr()

fig = plt.figure(figsize=(10, 5))

ax = plt.scatter(data_2014_FT['BasePay'], data_2014_FT['Benefits'])

plt.ylabel('Benefits ($)')
plt.xlabel('Base Pay ($)')

plt.show()

pal = sns.diverging_palette(0, 255, n=2)
ax = sns.factorplot(x='BasePay', y='Job_Group', hue='Gender', data=data_2014_FT,
                   size=10, kind="bar", palette=pal, ci=None)


plt.title('Full Time Workers')
plt.xlabel('Base Pay ($)')
plt.ylabel('Job Group')
plt.show()

salaries_by_group = pd.pivot_table(data = data_2014_FT, 
                                   values = 'BasePay',
                                   columns = 'Job_Group', index='Gender',
                                   aggfunc = np.mean)

count_by_group = pd.pivot_table(data = data_2014_FT, 
                                   values = 'Id',
                                   columns = 'Job_Group', index='Gender',
                                   aggfunc = len)

salaries_by_group

fig = plt.figure(figsize=(10, 15))
sns.set(font_scale=1.5)

differences = (salaries_by_group.loc['female'] - salaries_by_group.loc['male'])*100/salaries_by_group.loc['male']

labels  = differences.sort_values().index

x = differences.sort_values()
y = [i for i in range(len(differences))]
palette = sns.diverging_palette(240, 10, n=28, center ='dark')
ax = sns.barplot(x, y, orient = 'h', palette = palette)

#Draws the two arrows
bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="white", ec="black", lw=1)
t = plt.text(5.5, 12, "Higher pay for females", ha="center", va="center", rotation=0,
            size=15,
            bbox=bbox_props)
bbox_props2 = dict(boxstyle="larrow,pad=0.3", fc="white", ec="black", lw=1)
t = plt.text(-5.5, 12, "Higher pay for males", ha="center", va="center", rotation=0,
            size=15,
            bbox=bbox_props2)

#Labels each bar with the percentage of females
percent_labels = count_by_group[labels].iloc[0]*100                 /(count_by_group[labels].iloc[0] + count_by_group[labels].iloc[1])
for i in range(len(ax.patches)):
    p = ax.patches[i]
    width = p.get_width()*1+1
    ax.text(15,
            p.get_y()+p.get_height()/2+0.3,
            '{:1.0f}'.format(percent_labels[i])+' %',
            ha="center") 
    ax.text(15, -1+0.3, 'Female Representation',
            ha="center", fontname='Arial', rotation = 0) 

    
plt.yticks(range(len(differences)), labels)
plt.title('Full Time Workers (Base Pay)')
plt.xlabel('Mean Percent Difference in Pay (Females - Males)')
plt.xlim(-11, 11)
plt.show()

contingency_table = pd.crosstab(
    data_2014_FT['Gender'],
    data_2014_FT['Job_Group'],
    margins = True
)
contingency_table

#Assigns the frequency values
femalecount = contingency_table.iloc[0][0:-1].values
malecount = contingency_table.iloc[1][0:-1].values

totals = contingency_table.iloc[2][0:-1]
femalepercentages = femalecount*100/totals
malepercentages = malecount*100/totals


malepercentages=malepercentages.sort_values(ascending=True)
femalepercentages=femalepercentages.sort_values(ascending=False)
length = range(len(femalepercentages))

#Plots the bar chart
fig = plt.figure(figsize=(10, 12))
sns.set(font_scale=1.5)
p1 = plt.barh(length, malepercentages.values, 0.55, label='Male', color='#AFAFF5')
p2 = plt.barh(length, femalepercentages, 0.55, left=malepercentages, color='#EFAFB5', label='Female')



labels = malepercentages.index
plt.yticks(range(len(malepercentages)), labels)
plt.xticks([0, 25, 50, 75, 100], ['0 %', '25 %', '50 %', '75 %', '100 %'])
plt.xlabel('Percentage of Males')
plt.title('Gender Representation (San Francisco)')
plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc=3,
           ncol=2, mode="expand", borderaxespad=0)
plt.show()

from scipy import stats

#Significance testing by job title
job_titles = data_2014['JobTitle'].value_counts(dropna=True)
job_titles_over_100 = job_titles[job_titles > 100 ]

t_scores = {}

for title,count in job_titles_over_100.iteritems():
    male_pay = pd.to_numeric(male_only[male_only['JobTitle'] == title]['BasePay'])
    female_pay = pd.to_numeric(female_only[female_only['JobTitle'] == title]['BasePay'])
    
    if female_pay.shape[0] < 30:
        continue
    if male_pay.shape[0] < 30:
        continue

    t_scores[title] = stats.ttest_ind_from_stats(       
        mean1=male_pay.mean(), std1=(male_pay.std()), nobs1= male_pay.shape[0], \
        mean2=female_pay.mean(), std2=(female_pay.std()), nobs2=female_pay.shape[0], \
        equal_var=False)
    
for key, value in t_scores.items():
    if value[1] < 0.05:
        print(len(key)*'-')        
        print(key)
        print(len(key)*'-')
        print(t_scores[key])
        print(' ')
        print('Male: {}'.format((male_only[male_only['JobTitle'] == key]['BasePay']).mean()))
        print('sample size: {}'.format(male_only[male_only['JobTitle'] == key].shape[0]))
        print(' ')
        print('Female: {}'.format((female_only[female_only['JobTitle'] == key]['BasePay']).mean()))
        print('sample size: {}'.format(female_only[female_only['JobTitle'] == key].shape[0]))

len(t_scores)

#Reads in the data
nb_data = pd.read_csv('newport-beach-2016.csv')

#Creates job groups
def find_job_title_nb(row):
    titles = collections.OrderedDict([
        ('Police',['police', 'sherif', 'probation', 'sergeant', 'officer', 'lieutenant']),
        ('Fire', ['fire']),
        ('Transit',['mta', 'transit']),
        ('Medical',['anesth', 'medical', 'nurs', 'health', 'physician', 'orthopedic', 'pharm', 'care']),
        ('Architect', ['architect']),
        ('Court',['court', 'legal']),
        ('Mayor Office', ['mayoral']),
        ('Library', ['librar']),
        ('Public Works', ['public']),
        ('Attorney', ['attorney']),
        ('Custodian', ['custodian']),
        ('Gardener', ['garden']),
        ('Recreation Leader', ['recreation']),
        ('Automotive',['automotive', 'mechanic', 'truck']),
        ('Engineer',['engineer', 'engr', 'eng', 'program']),
        ('General Laborer',['general laborer', 'painter', 'inspector', 'carpenter', 'electrician', 'plumber', 'maintenance']),
        ('Food Services', ['food serv']),
        ('Clerk', ['clerk']),
        ('Porter', ['porter']),
        ('Airport Staff', ['airport']),
        ('Social Worker',['worker']),        
        ('Guard', ['guard']),
        ('Assistant',['aide', 'assistant', 'secretary', 'attendant']),        
        ('Analyst', ['analy']),
        ('Manager', ['manager'])      
    ])       
         
    #Loops through the dictionaries
    for group, keywords in titles.items():
        for keyword in keywords:
            if keyword in row.lower():
                return group
    return 'Other'

start_time = time.time()    
nb_data["Job_Group"]=data["JobTitle"].map(find_job_title_nb)

#Create the 'Gender' column based on employee's first name.
d = gender.Detector(case_sensitive=False)
nb_data['FirstName'] = nb_data['Employee Name'].str.split().apply(lambda x: x[0])
nb_data['Gender'] = nb_data['FirstName'].apply(lambda x: d.get_gender(x))
nb_data['Gender'].value_counts()

#Retain data with 'male' and 'female' names.
nb_male_female_only = nb_data[(nb_data['Gender'] == 'male') | (nb_data['Gender'] == 'female')]
nb_male_female_only['Gender'].value_counts()

#Seperates full time/part time data
nb_data_FT = nb_male_female_only[nb_male_female_only['Status'] == 'FT']
nb_data_PT = nb_male_female_only[nb_male_female_only['Status'] == 'PT']

nb_data_FT.head()

fig = plt.figure(figsize=(10, 5))

nb_male_only = nb_data_PT[nb_data_PT['Gender'] == 'male']
nb_female_only = nb_data_PT[nb_data_PT['Gender'] == 'female']
ax = fig.add_subplot(1, 1, 1)
ax = sns.kdeplot(nb_male_only['Total Pay & Benefits'], color ='Blue', label='Male', shade=True)
ax = sns.kdeplot(nb_female_only['Total Pay & Benefits'], color='Red', label='Female', shade=True)
plt.title('Newport Beach: Part Time Workers')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay + Benefits ($)')
plt.xlim(0, 400000)
plt.yticks([])

plt.show()

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(hspace=.5)  

#Generate the top chart
nb_male_only = nb_data_FT[nb_data_FT['Gender'] == 'male']
nb_female_only = nb_data_FT[nb_data_FT['Gender'] == 'female']
ax = fig.add_subplot(2, 1, 1)
ax = sns.kdeplot(nb_male_only['Total Pay & Benefits'], color ='Blue', label='Male', shade=True)
ax = sns.kdeplot(nb_female_only['Total Pay & Benefits'], color='Red', label='Female', shade=True)
plt.title('Newport Beach: Full Time Workers')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay + Benefits ($)')
plt.xlim(0, 400000)
plt.yticks([])

#Generate the bottom chart
male_only = data_2014_FT[data_2014_FT['Gender'] == 'male']
female_only = data_2014_FT[data_2014_FT['Gender'] == 'female']
ax2 = fig.add_subplot(2, 1, 2)
ax2 = sns.kdeplot(male_only['TotalPayBenefits'], color ='Blue', label='Male', shade=True)
ax2 = sns.kdeplot(female_only['TotalPayBenefits'], color='Red', label='Female', shade=True)
plt.title('San Francisco: Full Time Workers')
plt.ylabel('Density of Employees')
plt.xlabel('Total Pay + Benefits ($)')
plt.xlim(0, 400000)
plt.yticks([])

plt.show()

nb_salaries_by_group = pd.pivot_table(data = nb_data_FT, 
                                   values = 'Base Pay',
                                   columns = 'Job_Group', index='Gender',
                                   aggfunc = np.mean,)

nb_salaries_by_group

fig = plt.figure(figsize=(10, 7.5))
sns.set(font_scale=1.5)

differences = (nb_salaries_by_group.loc['female'] - nb_salaries_by_group.loc['male'])*100/nb_salaries_by_group.loc['male']
nb_labels  = differences.sort_values().index
x = differences.sort_values()
y = [i for i in range(len(differences))]
nb_palette = sns.diverging_palette(240, 10, n=9, center ='dark')
ax = sns.barplot(x, y, orient = 'h', palette = nb_palette)


plt.yticks(range(len(differences)), nb_labels)
plt.title('Newport Beach: Full Time Workers (Base Pay)')
plt.xlabel('Mean Percent Difference in Pay (Females - Males)')
plt.xlim(-25, 25)
plt.show()

nb_contingency_table = pd.crosstab(
    nb_data_FT['Gender'],
    nb_data_FT['Job_Group'],
    margins = True
)
nb_contingency_table

#Assigns the frequency values
nb_femalecount = nb_contingency_table.iloc[0][0:-1].values
nb_malecount = nb_contingency_table.iloc[1][0:-1].values

nb_totals = nb_contingency_table.iloc[2][0:-1]
nb_femalepercentages = nb_femalecount*100/nb_totals
nb_malepercentages = nb_malecount*100/nb_totals


nb_malepercentages=nb_malepercentages.sort_values(ascending=True)
nb_femalepercentages=nb_femalepercentages.sort_values(ascending=False)
nb_length = range(len(nb_malepercentages))

#Plots the bar chart
fig = plt.figure(figsize=(10, 12))
sns.set(font_scale=1.5)
p1 = plt.barh(nb_length, nb_malepercentages.values, 0.55, label='Male', color='#AFAFF5')
p2 = plt.barh(nb_length, nb_femalepercentages, 0.55, left=nb_malepercentages, color='#EFAFB5', label='Female')
labels = nb_malepercentages.index
plt.yticks(range(len(nb_malepercentages)), labels)
plt.xticks([0, 25, 50, 75, 100], ['0 %', '25 %', '50 %', '75 %', '100 %'])
plt.xlabel('Percentage of Males')
plt.title('Gender Representation (Newport Beach)')
plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc=3,
           ncol=2, mode="expand", borderaxespad=0)
plt.show()

fig = plt.figure(figsize=(10, 5))

colors = ['#AFAFF5', '#EFAFB5']
labels = ['Male', 'Female']
sizes = [len(nb_male_only), len(nb_female_only)]
explode = (0.05, 0)
sns.set(font_scale=1.5)
ax = fig.add_subplot(1, 2, 1)
ax = plt.pie(sizes, labels=labels, explode=explode, colors=colors, shadow=True, startangle=90, autopct='%1.f%%')
plt.title('Newport Beach: Full Time')


sizes = [len(male_only), len(female_only)]
explode = (0.05, 0)
sns.set(font_scale=1.5)
ax2 = fig.add_subplot(1, 2, 2)
ax2 = plt.pie(sizes, labels=labels, explode=explode, colors=colors, shadow=True, startangle=90, autopct='%1.f%%')
plt.title('San Francisco: Full Time')

plt.show()



