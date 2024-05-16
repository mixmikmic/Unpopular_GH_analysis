import numpy as np
import pandas as pd
import zipfile
pd.set_option('display.max_columns', 500)

# list of bay area county FIPS codes
bay_area_cfips = [1,13,41,55,75,81,85,95,97]

# load household records
z = zipfile.ZipFile('../data/csv_hca_2013_5yr.zip')
df1 = pd.read_csv(z.open('ss13hca.csv'))
print len(df1)

# limit to bay area counties
cfips = np.floor(df1.PUMA00/100) # county fips
df_h = df1[cfips.isin(bay_area_cfips)]
print len(df_h)

# load person records
z = zipfile.ZipFile('../data/csv_pca_2013_5yr.zip')
df2 = pd.read_csv(z.open('ss13pca.csv'))
print len(df2)

# limit to bay area and heads of household
cfips = np.floor(df2.PUMA00/100) # county fips
df_p = df2[cfips.isin(bay_area_cfips) & (df2.RELP == 0)]
print len(df_p)

# HOUSEHOLD RECORDS
# TEN is tenure: 1 and 2 = owned, 3 = rented

# PERSON RECORDS
# RAC1P is race code: 1 = white, 2 = black, 6 = asian
# HISP is hispanic code: >1 = hispanic

# merge and discard unneeded columns
df = df_h[['SERIALNO','TEN']].merge(df_p[['SERIALNO','RAC1P','HISP']], on='SERIALNO')
print len(df_p)

# rename to lowercase for consistency with urbansim
df.columns = [s.lower() for s in df.columns.values]

# set index and fix data types
df = df.set_index('serialno')
df['ten'] = df.ten.astype(int)
print df.head()

# save to data folder
df.to_csv('../data/household_extras.csv')

# List of year-2000 SuperPUMAs in the Bay Area, from here:
#   https://usa.ipums.org/usa/resources/volii/maps/ca_puma5.pdf

ba_puma1 = [
    40,  # Sonoma, Marin
    50,  # Napa, Solano
    121, # Contra Costa 
    122,
    130, # San Francisco
    140, # San Mateo
    151, # Alameda
    152, 
    153,
    161, # Santa Clara
    162,
    163]

# Read raw PUMS text file

# Helpful links for layouts and series definitions: 
#   http://www.census.gov/prod/cen2000/doc/pums.pdf
#   http://www.census.gov/population/cen2000/pumsrec5p.xls
#   http://www.census.gov/support/pums.html

# Variables to save from household and person records
h_serialno = []
h_puma1 = []    # latter three digits of SuperPUMA (first two refer to state) 
p_serialno = []
p_white = []    # 1 = white, alone or in combination with other races (vs 0)
p_black = []    # 1 = black, alone or in combination with other races (vs 0)
p_asian = []    # 1 = asian, alone or in combination with other races (vs 0)
p_hisp = []     # 1 = hispanic or latino origin (vs 0)

# Hispanic origin is recoded here from the HISPAN field, which doesn't match the format
# of the others and has been renamed in newer PUMS records anyway. 
#   HISPAN=1 => P_HISP=0
#   HISPAN>1 => P_HISP=1

with zipfile.ZipFile('../data/PUMS_2000_5yr_CA.zip') as z:
    with z.open('PUMS5_06.TXT') as f:
        for line in f:
            record_type = line[0]  # 'H' or 'P'
            if (record_type == 'H'):
                h_serialno.append(int(line[1:8]))
                h_puma1.append(int(line[20:23]))
            if (record_type == 'P'):
                relationship = line[16:18]  # head of household (01), etc
                if (relationship == '01'):
                    p_serialno.append(int(line[1:8]))
                    p_white.append(int(line[31]))
                    p_black.append(int(line[32]))
                    p_asian.append(int(line[34]))
                    hispan = int(line[27:29])
                    p_hisp.append(1 if (hispan > 1) else 0)

print "%d households" % len(h_serialno)
print len(h_puma1)
print "%d persons" % len(p_serialno)
print len(p_white)
print len(p_black)
print len(p_asian)
print len(p_hisp)

df_h = pd.DataFrame.from_dict({
        'serialno': h_serialno, 
        'puma1': h_puma1 })

df_p = pd.DataFrame.from_dict({
        'serialno': p_serialno,
        'white': p_white,
        'black': p_black,
        'asian': p_asian,
        'hisp': p_hisp })

# print df_h.describe()
# print df_p.describe()

# Merge dataframes and discard if outside the bay area

df = df_h.merge(df_p, on='serialno')
df = df[df.puma1.isin(ba_puma1)]
print df.describe()

# Set index

df = df.set_index('serialno')
print df.head()

# Save to data folder

df.to_csv('../data/household_extras.csv')









df.puma1.dtype

