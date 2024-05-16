## Loading dataset ##

import pandas as pd

# Set index_col to False to avoid pandas thinking that the first column is row indexes (it's age).
income = pd.read_csv("income.csv", index_col=False)
print(income.head(2))

## Converting categorical variables ##

# Convert a single column from text categories into numbers.
col = pd.Categorical(income["workclass"])
income["workclass"] = col.codes
print(income["workclass"].head(5))

cols = ['education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'high_income']
for c in cols:
    income[c] = pd.Categorical(income[c]).codes

print(income.head(2))

private_incomes = income[income['workclass'] == 4]
public_incomes = income[income['workclass'] != 4]
print(private_incomes.shape, public_incomes.shape)

import math

length = income.shape[0]
prob_high = sum(income['high_income'] == 1) / length
prob_low = sum(income['high_income'] == 0) / length
income_entropy = -(prob_high * math.log(prob_high, 2) + prob_low * math.log(prob_low, 2))
print(income_entropy)

# General Function to get entropy
import numpy as np
def calc_entropy(column):
    """
    Calculate entropy given a pandas Series, list, or numpy array.
    """
    # Compute the counts of each unique value in the column.
    counts = np.bincount(column)
    # Divide by the total column length to get a probability.
    probabilities = counts / len(column)
    
    # Initialize the entropy to 0.
    entropy = 0
    # Loop through the probabilities, and add each one to the total entropy.
    for prob in probabilities:
        if prob > 0:
            entropy += prob * math.log(prob, 2)
    
    return -entropy

calc_entropy(income['high_income'])

# Get the median
age_median = income['age'].median()

# left and right split
left_split = income[income['age'] <= age_median]['high_income']
right_split = income[income['age'] > age_median]['high_income']

# obtain the entropy
income_entropy = calc_entropy(income['high_income'])

# Information gain
age_information_gain = income_entropy - (left_split.shape[0] / income.shape[0] * calc_entropy(left_split) + right_split.shape[0] / income.shape[0] * calc_entropy(right_split))
print(age_information_gain)

# General function to get information gain
def calc_information_gain(data, split_name, target_name):
    """
    Calculate information gain given a dataset, column to split on, and target.
    """
    # Calculate original entropy.
    original_entropy = calc_entropy(data[target_name])
    
    # Find the median of the column we're splitting.
    column = data[split_name]
    median = column.median()
    
    # Make two subsets of the data based on the median.
    left_split = data[column <= median]
    right_split = data[column > median]
    
    # Loop through the splits, and calculate the subset entropy.
    to_subtract = 0
    for subset in [left_split, right_split]:
        prob = (subset.shape[0] / data.shape[0]) 
        to_subtract += prob * calc_entropy(subset[target_name])
    
    # Return information gain.
    return original_entropy - to_subtract

print(calc_information_gain(income, "age", "high_income"))

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import re
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 14})


# Function to find the column in columns to split on
def find_best_column(data, target_name, columns, iplot=0):
    # data is a dataframe
    # target_name is the name of the target variable
    # columns is a list of potential columns to split on
    information_gains = [calc_information_gain(data, c, target_name) for c in columns]
    
    # plot data optional
    if iplot==1:
        plt.figure(figsize=(25,5))
        x_pos = np.arange(len(columns))
        plt.bar(x_pos, information_gains,align='center', alpha=0.5)
        plt.xticks(x_pos, columns)
        plt.ylabel('information gains')
        plt.show()
    
    # return column name with highest gain
    highest_gain = columns[information_gains.index(max(information_gains))] 
    return highest_gain

# A list of columns to potentially split income with
columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

income_split = find_best_column(income, 'high_income', columns, iplot=1)
print(income_split, "has highest gain")

# Create a dictionary to hold the tree  
# It has to be outside of the function so we can access it later
tree = {}

# This list will let us number the nodes  
# It has to be a list so we can access it inside the function
nodes = []

def id3(data, target, columns, tree):
    unique_targets = pd.unique(data[target])
    
    # Assign the number key to the node dictionary
    nodes.append(len(nodes) + 1)
    tree["number"] = nodes[-1]

    if len(unique_targets) == 1:
        # assign "label" field to  node dictionary
        if unique_targets[0]==1:
            tree["label"] = 1
        else: 
            tree["label"] = 0
        return
    
    best_column = find_best_column(data, target, columns)
    column_median = data[best_column].median()
    
    # assign "column", "median" to node dictionary
    tree["column"] = best_column
    tree["median"] = column_median
    
    left_split = data[data[best_column] <= column_median]
    right_split = data[data[best_column] > column_median]
    split_dict = [["left", left_split], ["right", right_split]]
    
    for name, split in split_dict:
        tree[name] = {}
        id3(split, target, columns, tree[name])

# Create the data set that we used in the example on the last screen
data = pd.DataFrame([
    [0,20,0],
    [0,60,2],
    [0,40,1],
    [1,25,1],
    [1,35,2],
    [1,55,1]
    ])
# Assign column names to the data
data.columns = ["high_income", "age", "marital_status"]

# Call the function on our data to set the counters properly
id3(data, "high_income", ["age", "marital_status"], tree)

print(tree)

def print_with_depth(string, depth):
    # Add space before a string
    prefix = "    " * depth
    # Print a string, and indent it appropriately
    print("{0}{1}".format(prefix, string))
    
    
def print_node(tree, depth):
    # Check for the presence of "label" in the tree
    if "label" in tree:
        # If found, then this is a leaf, so print it and return
        print_with_depth("Leaf: Label {0}".format(tree["label"]), depth)
        # This is critical
        return
    # Print information about what the node is splitting on
    print_with_depth("{0} > {1}".format(tree["column"], tree["median"]), depth)
    
    # Create a list of tree branches
    branches = [tree["left"], tree["right"]]
        
    # recursively call print_node on each branch, increment depth
    print_node(branches[0], depth+1)
    print_node(branches[1], depth+1)
print_node(tree, 0)

def predict(tree, row):
    if "label" in tree:
        return tree["label"]
    
    column = tree["column"]
    median = tree["median"]
    # If row[column] is <= median,
    # return the result of prediction on left branch of tree
    # else right
    
    if row[column] <= median:
        return predict(tree["left"], row)
    else:
        return predict(tree["right"], row)

print(predict(tree, data.iloc[0]))

# Making Multiple Predictions using pandas apply method
# df.apply(func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds)

new_data = pd.DataFrame([
    [40,0],
    [20,2],
    [80,1],
    [15,1],
    [27,2],
    [38,1]
    ])
# Assign column names to the data
new_data.columns = ["age", "marital_status"]

def batch_predict(tree, df):
    # Insert your code here
    return df.apply(lambda x: predict(tree, x), axis=1)
    

predictions = batch_predict(tree, new_data)
print(predictions)

