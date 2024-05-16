import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
get_ipython().run_line_magic('matplotlib', 'inline')

wellcome_trust_raw = pd.read_csv('WELLCOME_APCspend2013_forThinkful.csv', encoding='ISO-8859-1')
wellcome_trust = pd.read_csv('WELLCOME_APCspend2013_forThinkful.csv', encoding='ISO-8859-1')

wellcome_trust.shape

wellcome_trust.head(5)

#Since we are looking specifically at Journal title, Article title, and Cost, want to drop NaN's there.

wellcome_trust = wellcome_trust.dropna(subset=['Journal title','Article title', 'COST (£) charged to Wellcome (inc VAT when charged)'])

#Check to see missing values per column - confirm if our dropna() above worked.

def missing(x):
  return sum(x.isnull())

print("Missing values per column:")
print(wellcome_trust.apply(missing, axis=0))

#Make all journal titles uppercase to avoid mismatches
wellcome_trust['Journal title clean'] = wellcome_trust['Journal title'].apply(lambda x: str(x).upper())

#Similarly, strip whitespace from all journal titles to avoid mismatches
wellcome_trust['Journal title clean'] = wellcome_trust['Journal title clean'].apply(lambda x: str(x).strip())

#Show in alphabetical order to compare against value_counts() below
wellcome_trust.sort_values(by='Journal title clean')

#Show value_counts and compare with above in order to determine if any need to be combined
pd.set_option("display.max_rows",3000)
wellcome_trust['Journal title clean'].value_counts().head(100)

#Combining categories

wellcome_trust['Journal title clean'] = wellcome_trust['Journal title clean'].replace(['PLOSONE', 'PLOS 1', 'PNAS', 'NEUROIMAGE: CLINICAL'], 
                                                                            ['PLOS ONE', 'PLOS ONE', 'PROCEEDINGS OF THE NATIONAL ACADEMY OF SCIENCES', 'NEUROIMAGE'])

#Now that we've combined, get our top 5 journals
wellcome_trust['Journal title clean'].value_counts().head(5)

#Create dataframe with only top 5 journals
wellcome_trust_top_journals = wellcome_trust[wellcome_trust["Journal title clean"].isin(['PLOS ONE', 'JOURNAL OF BIOLOGICAL CHEMISTRY', 'NEUROIMAGE', 'PROCEEDINGS OF THE NATIONAL ACADEMY OF SCIENCES', 'NUCLEIC ACIDS RESEARCH'])]

#Now calculate total articles for each
wellcome_trust_top_journals[['Journal title clean', 'Article title']].groupby('Journal title clean').count()

#Strip whitespace from the cost field
wellcome_trust['COST (£) charged to Wellcome (inc VAT when charged)'] = wellcome_trust['COST (£) charged to Wellcome (inc VAT when charged)'].apply(lambda x: str(x).strip())

#Define function to strip £ symbol when there is one
def remove_pound_symbol(x):
    if x.find('£') != -1:
        return x[1:]
    else:
        return x

#Apply function to the cost column
wellcome_trust['COST (£) charged to Wellcome (inc VAT when charged)'] = wellcome_trust['COST (£) charged to Wellcome (inc VAT when charged)'].apply(remove_pound_symbol)

#Change cost column to float
wellcome_trust['COST (£) charged to Wellcome (inc VAT when charged)'] = wellcome_trust['COST (£) charged to Wellcome (inc VAT when charged)'].astype(float)

#Create dataframe with just top 5 journals, similar to above
wellcome_trust_top_journals_stats = wellcome_trust[wellcome_trust["Journal title clean"].isin(['PLOS ONE', 'JOURNAL OF BIOLOGICAL CHEMISTRY', 'NEUROIMAGE', 'PROCEEDINGS OF THE NATIONAL ACADEMY OF SCIENCES', 'NUCLEIC ACIDS RESEARCH'])]

#Calculate the mean
wellcome_trust_top_journals_stats[['Journal title clean', 'COST (£) charged to Wellcome (inc VAT when charged)']].groupby('Journal title clean').mean()

#Calculate the standard deviation
wellcome_trust_top_journals_stats[['Journal title clean', 'COST (£) charged to Wellcome (inc VAT when charged)']].groupby('Journal title clean').std()

#Calculate the median
wellcome_trust_top_journals_stats[['Journal title clean', 'COST (£) charged to Wellcome (inc VAT when charged)']].groupby('Journal title clean').median()

