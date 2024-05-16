import pandas as pd
import matplotlib.pyplot as plt

jeopardy = pd.read_csv('jeopardy.csv')
jeopardy.head(5)

print(jeopardy.columns)

jeopardy.columns = ['Show Number', 'Air Date', 'Round', 'Category', 'Value',
       'Question', 'Answer']
jeopardy.columns

import re
def lowercase_no_punct(string):
    lower = string.lower()
    punremoved = re.sub('[^A-Za-z0-9\s]','', lower)
    return punremoved

jeopardy['clean_question'] = jeopardy['Question'].apply(lowercase_no_punct)
jeopardy['clean_answer'] = jeopardy['Answer'].apply(lowercase_no_punct)

def punremovandtoint(string):
    punremoved = re.sub('[^A-Za-z0-9\s]','', string)
    try:
        integer = int(punremoved)
    except Exception:
        integer = 0
    return integer

jeopardy['clean_values'] = jeopardy['Value'].apply(punremovandtoint)

jeopardy['Air Date'] = pd.to_datetime(jeopardy['Air Date'])

jeopardy.head()

def cleaner(series):
    split_answer = series['clean_answer'].split(' ')
    split_question = series['clean_question'].split(' ')
    match_count = 0
    if "the" in split_answer:
        split_answer.remove('the')
    if len(split_answer) == 0:
        return 0
    for item in split_answer:
        if item in split_question:
            match_count +=1
    return match_count/len(split_answer)

jeopardy['answer_in_question'] = jeopardy.apply(cleaner, axis=1)
jeopardy['answer_in_question'].mean()

question_overlap = []
#a python set is an unordered list of items
terms_used = set()
for idx, row in jeopardy.iterrows():
    split_question = row['clean_question'].split(" ")     
    match_count = 0
    newlist = []
    for word in split_question:
        if len(word) >= 6:
            newlist.append(word)
    for word in newlist:
        if word in terms_used:
            match_count += 1
    for word in newlist:
        terms_used.add(word)
    if len(newlist) > 0:
        match_count = match_count/len(newlist)
    question_overlap.append(match_count)

jeopardy['question_overlap'] = question_overlap

jeopardy['question_overlap'].mean()

def highvalue(row):
    value = 0
    if row['clean_values'] > 800:
        value = 1
    return value

jeopardy['high_value'] = jeopardy.apply(highvalue, axis =1)

high_value_count = jeopardy[jeopardy['high_value'] == 1].shape[0]
low_value_count = jeopardy[jeopardy['high_value'] == 0].shape[0]

print(high_value_count)
low_value_count

def highlowcounts(word):
    low_count = 0
    high_count = 0 
    for idx, row in jeopardy.iterrows():
        if word in row['clean_question'].split(' '):
            if row["high_value"] == 1:
                high_count += 1
            else:
                low_count += 1   
    return high_count, low_count

observed_expected = []
comparison_terms = list(terms_used)[:5]
comparison_terms

for term in comparison_terms:
    observed_expected.append(highlowcounts(term))

observed_expected

chi_squared =[]
from scipy.stats import chisquare
import numpy as np
for lists in observed_expected:
    total = sum(lists)
    total_prop = total/jeopardy.shape[0]
    expected_high = total_prop * high_value_count
    expected_low = total_prop * low_value_count
    observed = np.array([lists[0], lists[1]])
    expected = np.array([expected_high, expected_low])
    chi_squared.append(chisquare(observed, expected))

chi_squared

