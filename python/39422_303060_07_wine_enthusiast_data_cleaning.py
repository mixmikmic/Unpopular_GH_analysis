import pandas as pd
import dill
from glob import glob
import re
import numpy as np

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# # Read in the entire list and sort it numerically
file_list = glob('../priv/pkl/06_wine_enthusiast_dot_com_data_*.pkl')
int_sorter = lambda x: int(re.search(r"""06_wine_enthusiast_dot_com_data_(.+).pkl""", x).group(1))
file_list = sorted(file_list, key=int_sorter)

full_list = np.arange(1,6530)
num_list = np.array([int_sorter(x) for x in file_list])

mask = np.invert(np.in1d(full_list, num_list))
print(sum(mask))
full_list[mask]

# Read in just the last 32 files to check them
# file_list = ! ls -tr ../pkl/06_wine_enthusiast_dot_com_data* | tail -n 32

# Load and combine the data for the list of files
combined_data = list()
for fil in file_list:
    
    with open(fil, 'r') as fh:
        
        data = dill.load(fh)
        
        for key in data.keys():
            
            dat = data[key]
            
            if isinstance(dat, pd.Series):
                dat['url'] = key[1]
                dat['list_url_no'] = key[0]
                combined_data.append(dat)
            else:
                print(key)
                
combined_df = pd.concat(combined_data, axis=1).T


print((combined_df.review.apply(lambda x: len(x))==0).sum())
print(combined_df.isnull().sum())

combined_df.shape

# Drop the ones without reviews
mask = combined_df.review.apply(lambda x: len(x)==0).pipe(np.invert)
combined_df = combined_df.loc[mask]
combined_df.shape

# Convert prices to floats, have to remove the 'buy now'
replace_string = combined_df.loc[combined_df.price.str.contains('Buy Now'), 
                                 'price'].unique()[0]
combined_df.loc[combined_df.price==replace_string, 'price'] = np.NaN

combined_df['price'] = combined_df.price.astype(float)

combined_df['rating'] = combined_df.rating.astype(float)

# There are some % alcohol values that are way too high (above 100), 
# set anything above 50% to NaN
mask = combined_df.alcohol!='N/A'
combined_df.loc[mask, 'alcohol'] = combined_df.loc[mask,'alcohol'].str.replace(r"""\s*%""",'')
combined_df.loc[mask.pipe(np.invert), 'alcohol'] = np.NaN
combined_df['alcohol'] = combined_df.alcohol.astype(float)

mask = combined_df.alcohol >= 40.0
combined_df.loc[mask, 'alcohol'] = np.NaN

# Fixing bottle size requires some more extensive work
mask_L = combined_df.bottle_size.str.contains('L')
mask_ml = combined_df.bottle_size.str.contains('ml')
mask_M = combined_df.bottle_size.str.contains('ML')

combined_df.loc[mask_M, 'bottle_size'] = (combined_df
                                           .loc[mask_M, 'bottle_size']
                                           .str.replace(r"""ML""",'')
                                           .astype(float))

combined_df.loc[mask_L, 'bottle_size'] = (combined_df
                                       .loc[mask_L, 'bottle_size']
                                       .str.replace(r"""\s*L""",'')
                                       .astype(float)*1000)

combined_df.loc[mask_ml, 'bottle_size'] = (combined_df
                                           .loc[mask_ml, 'bottle_size']
                                           .str.replace(r"""\s*ml""",'')
                                           .astype(float))

combined_df['bottle_size'] = combined_df.bottle_size.astype(float)

combined_df['date_published'] = pd.to_datetime(combined_df.date_published)

# Was cleaning user ratings, but I decided to discard them as not useful
# combined_df['user_avg_rating'] = combined_df.user_avg_rating.str.replace(r""" \[Add Your Review\]""",'').head()
# combined_df.user_avg_rating.unique()
del combined_df['user_avg_rating']

# Some reviewers sign their reviews--remove these initials
mask = combined_df.review.str.contains(r"""\s+-\s?[A-Z]\.[A-Z]\.$""")
combined_df.loc[mask,'review'] = (combined_df
                                  .loc[mask,'review']
                                  .str.replace(r"""\s+-\s?[A-Z]\.[A-Z]\.$""", ''))

combined_df['title'].head()

def convert_year(ser):
    try:
        return int(ser)
    except:
        return ser
    
combined_df['year'] = combined_df['title'].str.extract(r""" ((?:19|20)[0-9]{2}) """, expand=True).apply(convert_year)

combined_df.year.isnull().sum()

# Discard blends
mask = combined_df.variety.str.contains('Blend').astype(np.bool).pipe(np.invert)
combined_df = combined_df.loc[mask]

combined_df.shape

# Discard all types except White, Red, and Rose
combined_df.category.unique()

mask = combined_df.category.isin(['White', 'Red', 'Rose'])
combined_df = combined_df.loc[mask]
combined_df.shape

# Clean up of wine variety names
# Now rename  probematic class names

#### Untouched class names ####
# Barbera                                        1091
# Cabernet Franc                                 1733
# Cabernet Sauvignon                            15830
# Chardonnay                                    17800
# Chenin Blanc                                    767
# Malbec                                         3681
# Merlot                                         6699
# Petit Verdot                                    287
# Pinot Blanc                                     634
# Pinot Noir                                    18284
# Pinotage                                        323
# Portuguese Red                                 2721
# Portuguese White                               1136
# Tempranillo                                    2976
# Viognier                                       1689
# Zinfandel                                      5216
# Gamay                                          1028
# Grenache                                        768
# Nebbiolo                                       3212
# Riesling                                       6819
# Sangiovese                                     4193
# Sauvignon Blanc                                7650

rename_dict = {'Aglianico, Italian Red'         :   'Aglianico',
 'Albariño'                                     :   'Albarino',
 'Blaufränkisch, Other Red'                     :   'Blaufrankisch',
 'Carmenère'                                    :   'Carmenere',
 'Corvina, Rondinella, Molinara, Italian Red'   :   'Corvina',
 'Dolcetto, Italian Red'                        :   'Dolcetto',
 'Garganega, Italian White'                     :   'Garganega',
 'Garnacha, Grenache'                           :   'Grenache',
 'Gewürztraminer'                               :   'Gewurztraminer',
 'Gewürztraminer, Gewürztraminer'               :   'Gewurztraminer',
 'Grüner Veltliner'                             :   'Gruner Veltliner',
 'Melon, Other White'                           :   'Melon',
 'Montepulciano, Italian Red'                   :   'Montepulciano',
 'Mourvèdre'                                    :   'Mourvedre',
 "Nero d'Avola, Italian Red"                    :   "Nero d Avola",
 'Petite Sirah'                                 :   'Petite Syrah',
 'Pinot Grigio, Pinot Grigio/Gris'              :   'Pinot Grigio',
 'Pinot Gris, Pinot Grigio/Gris'                :   'Pinot Grigio',
 'Primitivo, Zinfandel'                         :   'Zinfandel',
 'Rosé'                                         :   'Rose',
 'Sangiovese Grosso, Sangiovese'                :   'Sangiovese',
 'Sauvignon, Sauvignon Blanc'                   :   'Sauvignon Blanc',
 'Shiraz, Shiraz/Syrah'                         :   'Syrah',
 'Syrah, Shiraz/Syrah'                          :   'Syrah',
 'Tinta de Toro, Tempranillo'                   :   'Tempranillo',
 'Torrontés'                                    :   'Torrontes',
 'Verdejo, Spanish White'                       :   'Verdejo',
 'Vermentino, Italian White'                    :   'Vermentino'}

def val_rename(val):
    if val in rename_dict.keys():
        return rename_dict[val]
    else:
        return val
    
combined_df['variety'] = combined_df.variety.apply(lambda x: val_rename(x))

wine_varieties = combined_df[['variety','category']].groupby(['category','variety']).size()
wine_varieties_vc = wine_varieties.sort_values(ascending=False).reset_index().rename(columns={0:'count'})
wine_varieties_vc = wine_varieties_vc.query("count>=1200")
wine_varieties_vc

20*1500

combined_df_sampled = list()

for idx,dat in wine_varieties_vc.iterrows():
    
    mask = (combined_df.category==dat.category)&(combined_df.variety==dat.variety)
    combined_df_cat = combined_df.loc[mask]
    
    if dat['count'] < 1500:
        index = combined_df_cat.index
    else:
        index = np.random.choice(combined_df_cat.index, 1500, replace=False)
        
    combined_df_sampled.append(combined_df_cat.loc[index])
        
combined_df_sampled = pd.concat(combined_df_sampled, axis=0)        

combined_df_sampled.groupby(['variety','category']).size()

combined_df_sampled.to_pickle('../priv/pkl/07_wine_enthusiast_data_small_cleaned.pkl')

# Keeping only those with at least 250 members cuts the number of varieties down to 50 but retains 93% of data
mask1 = wine_varieties>=500
wine_varieties.loc[mask1].shape, wine_varieties.loc[mask1].sum()/float(wine_varieties.sum())

print wine_varieties.loc[mask1].sum(), wine_varieties.sum(), wine_varieties.loc[mask1].nunique()
# print wine_varieties.sort_values(ascending=False).iloc[:wine_varieties.loc[mask1].nunique()]

wine_varieties = wine_varieties.loc[mask1]

mask2 = combined_df.variety.isin(wine_varieties.index.values.tolist())
combined_df_large_output = combined_df.loc[mask2]
print(combined_df_large_output.shape)

combined_df_large_output.variety.value_counts().sort_index()

combined_df.to_pickle('../priv/pkl/07_wine_enthusiast_data_cleaned.pkl')

