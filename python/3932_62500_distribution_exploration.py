from dsfuncs.processing import remove_outliers
from dsfuncs.dist_plotting import plot_var_dist, plot_binary_response
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def read_df(year): 
    """Read in a year's worth of data. 
    
    Args: 
        year: int
            Holds the year of data to read in. 
    """
    
    df = pd.read_csv('../../../data/csvs/detected_fires_MODIS_' + str(year) + '.csv', true_values =['t', 'True'], false_values=['f', 'False'])
    df.dropna(subset=['region_name'], inplace=True) # These will be obs. in Canada. 
    return df

# Test out my function and see what columns I actually want to look at the distributions of. 
fires_df_2012 = read_df(2012)
fires_df_2012.columns

fires_df_2012.info()

# I'm going to look at the following set of continous and categorical variables. 
continous_vars = ('lat', 'long', 'gmt', 'temp', 'spix', 'tpix', 'conf', 'frp', 'county_aland', 'county_awater')
categorical_vars = ('urban_areas_bool', 'src', 'sat_src')

# Test out the outliers function to make sure it runs. 
print fires_df_2012['lat'].values.shape
print remove_outliers(fires_df_2012['lat']).shape

# Testing out the plot_var_dist function for a categorical variable. 
plot_var_dist(fires_df_2012['urban_areas_bool'], categorical=True)

# Testing out the plot_var_dist function for a continuous variable. 
f, axes = plt.subplots(1, 4)
plot_var_dist(fires_df_2012['lat'], categorical=False, ax=axes[0:])

def check_dists(year, continous_vars, categorical_vars): 
    """Plot the distributions of varaibles for the inputted year. 
    
    Read in the data for the inputted year. Then, take the inputted 
    variable names in the continous_vars and categorical_vars parameters, 
    and plot their distributions. Do this separately for observations 
    labeled as forest-fires and those labeled as non forest-fires. 
    
    Args: 
        year: int
            Holds the year of data to use for plotting. 
        continous_vars: tuple (or other iterable) of strings
            Holds the names of the continuous variables to use for plotting. 
        categorical_vars: tuple (or other iterable) of strings. 
            Holds the names of the categorical variables to use for plotting. 
    """
    
    df = read_df(year)
    fires = df.query('fire_bool == 0')
    non_fires = df.query('fire_bool == 1')
    print 'Continuous Vars'
    print '-' * 50
    for var in continous_vars: 
        print 'Variable: {} : Non-fires, then fires'.format(var)
        f, axes = plt.subplots(1, 8, figsize=(20, 5))
        plot_var_dist(fires[var], categorical=False, ax=axes[0:4], show=False)
        plot_var_dist(non_fires[var], categorical=False, ax=axes[4:], show=False)
        plt.show()
    print 'Categorical Vars'
    print '-' * 50
    for var in categorical_vars: 
        print 'Variable: {} : Non-fires, then fires'.format(var)
        f, axes = plt.subplots(1, 2)
        plot_var_dist(fires[var], categorical=True, ax=axes[0], show=False)
        plot_var_dist(non_fires[var], categorical=True, ax=axes[1], show=False)
        plt.show()

check_dists(2012, continous_vars, categorical_vars)

check_dists(2013, continous_vars, categorical_vars)

check_dists(2014, continous_vars, categorical_vars)

check_dists(2015, continous_vars, categorical_vars)

def add_land_water_ratio(df):
    """Add a land_water_ratio column to the inputted DataFrame. 
    
    Add a new variable to the inputted DataFrame that represents 
    the ratio of a counties land area to its total area (land plus water). 
    
    Args: 
        df: Pandas DataFrame
    
    Return: Pandas DataFrame
    """
    
    df.eval('land_water_ratio = county_aland / (county_aland + county_awater)')
    return df

def plot_land_water_ratio(year): 
    """Plot the land_water_ratio distribution for the inputted year. 
    
    For the inputted year, read in the data, add a land_water_ratio 
    variable to the DataFrame, and then plot it's distribution. 
    
    Args: 
        year: int
            Holds the year of data to plot. 
    """
    
    
    df = read_df(year)
    df = add_land_water_ratio(df)
    var = 'land_water_ratio'
    
    fires = df.query('fire_bool == 0')[var]
    non_fires = df.query('fire_bool == 1')[var]
    
    print 'Fires mean: {}'.format(fires.mean())
    print 'Non Fires mean: {}'.format(non_fires.mean())
    
    f, axes = plt.subplots(1, 8, figsize=(20, 5))
    plot_var_dist(fires, categorical=False, ax=axes[0:4], show=False)
    plot_var_dist(non_fires, categorical=False, ax=axes[4:], show=False)
    plt.show()

for year in xrange(2012, 2016): 
    print 'Year: {}'.format(str(year))
    print '-' * 50
    plot_land_water_ratio(2012)

for year in xrange(2012, 2016): 
    df = read_df(year)
    print 'Year {}'.format(str(year))
    print '-' * 50
    for category in categorical_vars: 
        print category
        plot_binary_response(df, category, 'fire_bool')

