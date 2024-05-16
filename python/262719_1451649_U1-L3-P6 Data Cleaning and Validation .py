import pandas as pd
import numpy as np

df = pd.read_csv('WELLCOME_APCspend2013_forThinkful.csv', encoding='latin_1')
df.columns

df.head()

# Originally I wanted to see how messy each individual title would be and turns out there were a lot of variations.
def obtain(data):
    titles = {}
    for i in data:
        if titles.get(i):
            titles[i] += 1
        else:
            titles[i] = 1 
    return titles 

alpha = (df['Journal title'].values)

print(obtain(alpha))

# Rename the titles appropriately

df['Journal title'] = df['Journal title'].replace([
    'PLOSONE', 'PLOS ONE', 'PLOS 1', 'PLOS','PLoS One','PLoS ONE'], 'PLOS ONE') 
df['Journal title'] = df['Journal title'].replace([
    'ACTA D', 'ACTA CRYSTALLOGRAPHICA SECTION D', 'ACTA CRYSTALLOGRAPHY D', 'ACTA CRYSTALLOGRAPHICA, SECTION D',
    'ACTA CRYSTALLOGRAPHICA SECTION D, BIOLOGICAL CRYSTALLOGRAPHY'], 
    'ACTA CRYSTALLOGRAPHICA SECTION D: BIOLOGICAL CRYSTALLOGRAPHY') 
df['Journal title'] = df['Journal title'].replace([
    'AMERICAN JNL EPIDEMIOLOGY'], 'AMERICAN JOURNAL OF EPIDEMIOLOGY') 
df['Journal title'] = df['Journal title'].replace([
    'AMERICAN JOURNAL OF MEDICAL GENETICS PART A'], 'AMERICAN JOURNAL OF MEDICAL GENETICS') 
df['Journal title'] = df['Journal title'].replace([
    'ANTIMICROBIAL AGENTS AND CHEMOTHERAPY', 'ANTIMICROBIAL AGFENTS AND CHEMOTHERAPY'], 
    'ANTIMICROBIAL AGENTS & CHEMOTHERAPY') 
df['Journal title'] = df['Journal title'].replace([
    'ANGEWANDE CHEMIE', 'ANGEWANDTE CHEMIE INTERNATIONAL EDITION','ANGEW CHEMS INT ED' ],
    'ANGEWANDTE CHEMIE') 
df['Journal title'] = df['Journal title'].replace([
    'BEHAVIOUR RESEARCH AND THERAPY'], 'BEHAVIOR RESEARCH & THERAPY') 
df['Journal title'] = df['Journal title'].replace([
    'BIOCHEM JOURNAL', 'BIOCHEMICAL JOURNALS'], 'BIOCHEMICAL JOURNAL') 
df['Journal title'] = df['Journal title'].replace([
    'BIOCHEM SOC TRANS'], 'BIOCHEMICAL SOCIETY TRANSACTIONS') 
df['Journal title'] = df['Journal title'].replace([
    'BRITISH JOURNAL OF OPHTHALMOLOGY'], 'BRITISH JOURNAL OF OPTHALMOLOGY') 
df['Journal title'] = df['Journal title'].replace([
    'CELL DEATH DIFFERENTIATION'], 'CELL DEATH & DIFFERENTIATION') 
df['Journal title'] = df['Journal title'].replace([
    'CHILD: CARE, HEALTH DEVELOPMENT'], 'CHILD: CARE, HEALTH & DEVELOPMENT') 
df['Journal title'] = df['Journal title'].replace(['CURR BIOL'], 'CURRENT BIOLOGY') 
df['Journal title'] = df['Journal title'].replace(['DEV. WORLD BIOETH'], 'DEVELOPING WORLD BIOETHICS')
df['Journal title'] = df['Journal title'].replace([
    'EUROPEAN CHILD AND ADOLESCENT PSYCHIATTY'], 'EUROPEAN CHILD & ADOLESCENT PSYCHIATRY') 
df['Journal title'] = df['Journal title'].replace(['FEBS J'], 'FEBS JOURNAL')
df['Journal title'] = df['Journal title'].replace(['HUM RESOUR HEALTH'], 'HUMAN RESOURCES FOR HEALTH')

# Change everything to lower case first then.
# Afterwards, get rid of the spaces to lower chances of dupes.
beta = df['Journal title']

print(beta)

# This is to check one more time if there is anything missing.
print(beta.unique())

#Get the 5 most common journals 

beta.value_counts().head(5)

# Lets change the name of the cost section to type less


df['Pounds'] = df['COST (£) charged to Wellcome (inc VAT when charged)']
df.head()

# Everything should be the same so lets, remove the '£' sign 

df['Pounds'] = df['Pounds'].str.replace('£', '')
# Tried turning pounds into integers earlier but saw an error so I decided to see if I could find anything. 
#Turns out there are dollar signs too, let's remove that next 
df.head(200)

# Remove the dollar signs and convert the str into int since typing .mean() earlier gave me an error 
# and told me it was a str

df['Pounds'] = df['Pounds'].str.replace('$', '')
gamma = df['Pounds']

delta = pd.to_numeric(gamma)
print(delta)

delta.describe()

# Let's try to scrub the outlier.

def neutral(data):
    result = []
    for i in data:
        if i > 10000:
            continue
        else:
            result.append(i)
    return result

n_delta = neutral(delta)

new_df = pd.DataFrame(np.array(n_delta).reshape(2077,1))

new_df.describe()



