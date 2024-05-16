#Import pandas and read the data.
import pandas as pd
data = pd.read_csv("thanksgiving.csv", encoding ="Latin-1")

#Print the column names.
col = data.columns
print(col)

#Prints the table structure(row x column)
print("rows, columns: "+str(data.shape))

#Outputs the first 5 rows of the dataframe
data.head()

print(data["Do you celebrate Thanksgiving?"].value_counts())

data = data[data["Do you celebrate Thanksgiving?"] == "Yes"]
data.shape

data["What is typically the main dish at your Thanksgiving dinner?"].value_counts()

data_only_tofurkey = data[data["What is typically the main dish at your Thanksgiving dinner?"] == "Tofurkey"]
Gravy_and_tofurkey = data_only_tofurkey["Do you typically have gravy?"]
Gravy_and_tofurkey.value_counts()

apple = data["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Apple"]
apple_isnull = apple.isnull()

pumpkin = data["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Pumpkin"]
pumpkin_isnull = pumpkin.isnull()

pecan = data["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Pecan"]
pecan_isnull = pecan.isnull()

did_not_eat_pies = apple_isnull & pumpkin_isnull & pecan_isnull 

did_not_eat_pies.value_counts()

#Converts the age column to an integer.
def convert_to_int(column):
    if pd.isnull(column) == True:
        return None
    if pd.isnull(column) == False:
        string = column.split(' ')[0]
        string = string.replace('+', '')
        return int(string)
    
int_age = data["Age"].apply(convert_to_int)
data["int_age"] = int_age

#Outputs statistical data of the column.
data["int_age"].describe()

data["How much total combined money did all members of your HOUSEHOLD earn last year?"].value_counts()

income_col = data["How much total combined money did all members of your HOUSEHOLD earn last year?"]

def convert_to_int_inc(column):
    if pd.isnull(column) == True:
        return None
    string = column.split(' ')[0]
    if 'Prefer' in string:
        return None
    else:
        string = string.replace('$', '')
        string = string.replace(',', '')
        return int(string)
    
data['int_income']  = income_col.apply(convert_to_int_inc)
data['int_income'].describe()

less_150k = data["int_income"] < 150000
less_150k_data = data[less_150k]

how_far = less_150k_data["How far will you travel for Thanksgiving?"]
how_far.value_counts()

more_150k = data["int_income"] > 150000
more_150k_data = data[more_150k]

how_far_150k_plus = more_150k_data["How far will you travel for Thanksgiving?"]
how_far_150k_plus.value_counts()

high_income_athome = 49/(49+25+16+12) 
low_income_athome = 281/(203+150+55+281)
print(high_income_athome)
print(low_income_athome)

data.pivot_table(
    #The index takes a series as an input and populates the rows of the spreadsheet
    index = "Have you ever tried to meet up with hometown friends on Thanksgiving night?",
    
    #The columns takes a series as an input and populates the columns with its values
    columns = 'Have you ever attended a "Friendsgiving?"',
    
    #The values we populate the matrix with, by default the values will be the mean
    values = 'int_age'
)

data.pivot_table(
    index = "Have you ever tried to meet up with hometown friends on Thanksgiving night?",
    columns = 'Have you ever attended a "Friendsgiving?"',
    values = 'int_income'
)



