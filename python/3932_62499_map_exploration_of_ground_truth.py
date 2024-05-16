import pandas as pd
import matplotlib.pyplot as plt
import fiona
import subprocess
import datetime
from dsfuncs.geo_plotting import USMapBuilder
from dsfuncs.dist_plotting import plot_var_dist, plot_binary_response
get_ipython().magic('matplotlib inline')

def read_df(year, days_ahead): 
    """This function will read in a year of data, and add a month column. 
    
    Args: 
        year: str
        days_ahead: int
            Used to determine what data to read in.  
        
    Return:
        Pandas DataFrame
    """
    output_df = pd.read_csv('../../../data/csvs/day{}/detected_fires_MODIS_'.format(days_ahead) + str(year) + '.csv', 
            parse_dates=['date'], true_values=['t'], false_values=['f'])
    output_df['month'] = output_df.date.apply(lambda dt: dt.strftime('%B'))
    output_df.dropna(subset=['region_name'], inplace=True) # These will be obs. in Canada. 
    return output_df
    
def grab_by_location(df, state_names, county_names=None): 
    """Grab the data for a specified inputted state and county. 
    
    Args: 
        df: Pandas DataFrame
        state: set (or iterable of strings)
            State to grab for plotting.
        county: set (or iterable of strings) (optional)
            County names to grab for plotting. If None, simply grab the 
            entire state to plot. 
            
    Return: 
        Pandas DataFrame
    """
    if county_names: 
        output_df = df.query('state_name in @state_names and county_name in @county_names')
    else: 
        output_df = df.query('state_name in @state_names')
    return output_df

def grab_by_date(df, months=None, dt=None): 
    """Grab the data for a set of specified months.
    
    Args: 
        df: Pandas DataFrame
        months: set (or iterable of strings)
    
    Return: 
        Pandas DataFrame
    """
    if months is not None: 
        output_df = df.query("month in @months")
    else: 
        split_dt = dt.split('-')
        year, month, dt = int(split_dt[0]), int(split_dt[1]), int(split_dt[2])
        match_dt = datetime.datetime(year, month, dt, 0, 0, 0)
        output_df = df.query('date == @match_dt')
    return output_df

def format_df(df): 
    """Format the data to plot it on maps. 
    
    This function will grab the latitude and longitude 
    columns of the DataFrame, and return those, along 
    with a third column that will be newly generated. This 
    new column will hold what color we want to use to plot 
    the lat/long coordinate - I'll use red for fire and 
    green for non-fire. 
    
    Args: 
        df: Pandas DataFrame
    
    Return: 
        numpy.ndarray
    """
    
    keep_cols = ['long', 'lat', 'fire_bool']
    intermediate_df = df[keep_cols]
    output_df = parse_fire_bool(intermediate_df)
    output_array = output_df.values
    return output_array

def parse_fire_bool(df): 
    """Parse the fire boolean to a color for plotting. 
    
    Args: 
        df: Pandas DataFrame
        
    Return: 
        Pandas DataFrame
    """
    
    # Plot actual fires red and non-fires green. 
    output_df = df.drop('fire_bool', axis=1)
    output_df['plotting_mark'] = df['fire_bool'].apply(lambda f_bool: 'ro' if f_bool == True else 'go')
    return output_df

def read_n_parse(year, state_names, county_names=None, months=None, plotting=False, dt=None, days_ahead=0): 
    """Read and parse the data for plotting.
    
    Args: 
        year: str
        state_names: set (or other iterable) of state names
            State names to grab for plotting. 
        county_names: set (or other iterable) of county names (optional)
            County names to grab for plotting. 
        months: months (or other iterable) of months (optional)
            Month names to grab for plotting. 
        plotting: bool 
            Whether or not to format the data for plotting. 
            
    Return: 
        Pandas DataFrame
    """
    
    fires_df = read_df(year, days_ahead)
    if state_names: 
        fires_df = grab_by_location(fires_df, state_names, county_names)
    
    if months or dt: 
        fires_df = grab_by_date(fires_df, months, dt)
    
    if plotting: 
        fires_df = format_df(fires_df)
    return fires_df

def grab_fire_perimeters(dt, st_name, st_abbrev, st_fips, county=False): 
    """Grab the fire perimter boundaries for a given state and year.
    
    Args: 
        dt: str
        st_name: str
            State name of boundaries to grab. 
        st_abbrev: str
            State abbreviation used for bash script. 
        st_fips: str
            State fips used for bash script. 
            
    Return: Shapefile features. 
    """
            
    # Run bash script to pull the right boundaries from Postgres. 
    subprocess.call("./grab_perim_boundary.sh {} {} {}".format(st_abbrev, st_fips, dt), shell=True)
    
    filepath = 'data/{}/{}_{}_2D'.format(st_abbrev, st_abbrev, dt)
    return filepath

def plot_county_dt(st_name, st_abbrev, st_fips, county_name, fires_dt=None, perims_dt=None, 
                  days_ahead=0): 
    """Plot all obs., along with the fire-perimeter boundaries, for a given county and date. 
    
    Read in the data for the inputted year and parse it to the given state/county and date. 
    Read in the fire perimeter boundaries for the given state/county and date, and parse 
    those. Plot it all. 
    
    Args: 
        st_name: str
            State name to grab for plotting. 
        st_abbrev: str
            State abbrevation used for the bash script. 
        st_fips: str
            State fips used for the base script. 
        county_name: str
            County name to grab for plotting. 
        dt: str
            Date to grab for plotting. 
    """
    
    year = fires_dt.split('-')[0]
    perims_dt = fires_dt if not perims_dt else perims_dt
    county_names = [county_name] if not isinstance(county_name, list) else county_name
    fires_data = read_n_parse(year, state_names=st_name, county_names=county_name, dt=fires_dt, plotting=True,
                             days_ahead=days_ahead)
    county_map = USMapBuilder('data/county_shapefiles_2014/cb_2014.shp', geo_level='County', 
                state_names=[st_name], county_names=county_names, figsize=(40, 20), border_padding=0.1)
    fire_boundaries = grab_fire_perimeters(perims_dt, st_name, st_abbrev, st_fips)
    try: 
        county_map.plot_boundary(fire_boundaries)
    except Exception as e:
        print e
    county_map.plot_points(fires_data, 4)
    plt.show()

for dt in xrange(1, 9): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', 'Trinity', dt, days_ahead=0)

for days_ahead in (0, 1, 3, 5, 7): 
    dt = '2015-08-0' + str(days_ahead + 1) if days_ahead < 10 else '2015-08-' + str(days_ahead + 1)
    print 'Days Ahead: {}'.format(days_ahead)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', 'Trinity', '2015-08-01', dt, days_ahead=days_ahead)
    plt.clf()

for year in xrange(2012, 2016): 
    print 'Year: {}'.format(year)
    print '-' * 50
    for days_forward in (0, 1, 3, 5, 7): 
        print 'Days Forward: {}'.format(days_forward)
        print '=' * 50
        df = read_df(year, days_forward)
        df.info()
        print '=' * 50

for year in xrange(2012, 2016): 
    print 'Year: {}'.format(year)
    print '-' * 50
    for days_forward in (0, 1, 3, 5, 7): 
        print 'Days Forward: {}'.format(days_forward)
        df = read_df(year, days_forward)
        print df.fire_bool.mean(), df.fire_bool.sum()
    print '\n' * 2

continous_vars = ('lat', 'long', 'gmt', 'temp', 'spix', 'tpix', 'conf', 'frp', 'county_aland', 'county_awater')
categorical_vars = ('urban_areas_bool', 'src', 'sat_src')

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
    dfs = []
    for days_forward in (0, 1, 3, 5, 7): 
        dfs.append(read_df(year, days_forward))
    
    fires = []
    non_fires = []
    for df in dfs: 
        fires.append(df.query('fire_bool == 1'))
        non_fires.append(df.query('fire_bool == 0'))
        
    print 'Continuous Vars'
    print '-' * 50
    for var in continous_vars: 
        print 'Variable: {} : Fires, then non-fires'.format(var)
        print '-' * 50
        for idx, df in enumerate(dfs): 
            f, axes = plt.subplots(1, 8, figsize=(20, 5))
            fire_var = fires[idx][var]
            non_fire_var = non_fires[idx][var]
            print fire_var.mean(), non_fire_var.mean()
            plot_var_dist(fire_var, categorical=False, ax=axes[0:4], show=False)
            plot_var_dist(non_fire_var, categorical=False, ax=axes[4:], show=False)
            plt.show()
    print 'Categorical Vars'
    print '-' * 50
    for var in categorical_vars: 
        print 'Variable: {} : Fires, then non-fires'.format(var)
        print '-' * 50
        for idx, df in enumerate(dfs):
            f, axes = plt.subplots(1, 2)
            plot_var_dist(fires[idx][var], categorical=True, ax=axes[0], show=False)
            plot_var_dist(non_fires[idx][var], categorical=True, ax=axes[1], show=False)
            plt.show()

check_dists(2012, continous_vars, categorical_vars)

check_dists(2013, continous_vars, categorical_vars)

check_dists(2014, continous_vars, categorical_vars)

check_dists(2015, continous_vars, categorical_vars)

def read_df(year, meters_nearby=None): 
    """This function will read in a year of data, and add a month column. 
    
    Args: 
        year: str
        meters_nearby (optional): int
            How far we go out from the original boundaries to label
            forest-fires. Used to determine what data to read in. If 
            not passed in, use the original, raw data (stored in the days0 
            folder). 
        
    Return:
        Pandas DataFrame
    """
    if not meters_nearby: 
        output_df = pd.read_csv('../../../data/csvs/day0/detected_fires_MODIS_' + str(year) + '.csv', 
                parse_dates=['date'], true_values=['t'], false_values=['f'])
    else: 
        output_df = pd.read_csv('../../../data/csvs/fires_{}m/detected_fires_MODIS_'.format(meters_nearby) 
                        + str(year) + '.csv', parse_dates=['date'], true_values=['t'], 
                        false_values=['f'])
    output_df['month'] = output_df.date.apply(lambda dt: dt.strftime('%B'))
    output_df.dropna(subset=['region_name'], inplace=True) # These will be obs. in Canada. 
    return output_df

def read_n_parse(year, state_names, county_names=None, months=None, plotting=False, 
                 dt=None, meters_nearby=0): 
    """Read and parse the data for plotting.
    
    Args: 
        year: str
        state_names: set (or other iterable) of state names
            State names to grab for plotting. 
        county_names: set (or other iterable) of county names (optional)
            County names to grab for plotting. 
        months: months (or other iterable) of months (optional)
            Month names to grab for plotting. 
        plotting: bool 
            Whether or not to format the data for plotting. 
        meters_nearby: int
            How many meters to go out when labeleing fires. Used to figure out
            what data to read in. 
            
    Return: 
        Pandas DataFrame
    """
    
    fires_df = read_df(year, meters_nearby)
    if state_names: 
        fires_df = grab_by_location(fires_df, state_names, county_names)
    
    if months or dt: 
        fires_df = grab_by_date(fires_df, months, dt)
    
    if plotting: 
        fires_df = format_df(fires_df)
    return fires_df

def plot_county_dt(st_name, st_abbrev, st_fips, county_name, fires_dt=None, perims_dt=None, 
                  meters_nearby=0): 
    """Plot all obs., along with the fire-perimeter boundaries, for a given county and date. 
    
    Read in the data for the inputted year and parse it to the given state/county and date. 
    Read in the fire perimeter boundaries for the given state/county and date, and parse 
    those. Plot it all. 
    
    Args: 
        st_name: str
            State name to grab for plotting. 
        st_abbrev: str
            State abbrevation used for the bash script. 
        st_fips: str
            State fips used for the base script. 
        county_name: str
            County name to grab for plotting. 
        dt: str
            Date to grab for plotting. 
        meters_nearby: int
            Holds the number of meters to go out to look for fires. Used to 
            figure out what data to read in. 
    """
    
    year = fires_dt.split('-')[0]
    perims_dt = fires_dt if not perims_dt else perims_dt
    county_names = [county_name] if not isinstance(county_name, list) else county_name
    fires_data = read_n_parse(year, state_names=st_name, county_names=county_name, dt=fires_dt, plotting=True,
                             meters_nearby=meters_nearby)
    county_map = USMapBuilder('data/county_shapefiles_2014/cb_2014.shp', geo_level='County', 
                state_names=[st_name], county_names=county_names, figsize=(40, 20), border_padding=0.1)
    fire_boundaries = grab_fire_perimeters(perims_dt, st_name, st_abbrev, st_fips)
    try: 
        county_map.plot_boundary(fire_boundaries)
    except Exception as e:
        print e
    county_map.plot_points(fires_data, 4)
    plt.show()

# Back to the original for a second. 
for dt in xrange(1, 9): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', 'Trinity', dt)

# Let's look at 100m out. 
for dt in xrange(1, 9): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', 'Trinity', dt, meters_nearby=100)

# Let's look at 250m out. 
for dt in xrange(1, 9): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', 'Trinity', dt, meters_nearby=250)

# Let's look at 500m out. 
for dt in xrange(1, 9): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', 'Trinity', dt, meters_nearby=500)

for year in xrange(2012, 2016): 
    print 'Year: {}'.format(year)
    print '-' * 50
    for meters_nearby in (0, 100, 250, 500): 
        print 'Meters Nearby: {}'.format(meters_nearby)
        print '=' * 50
        df = read_df(year, meters_nearby)
        df.info()
        print '=' * 50

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
    dfs = []
    for meters_ahead in (0, 100, 250, 500): 
        dfs.append(read_df(year, meters_ahead))
    
    fires = []
    non_fires = []
    for df in dfs: 
        fires.append(df.query('fire_bool == 1'))
        non_fires.append(df.query('fire_bool == 0'))
        
    print 'Continuous Vars'
    print '-' * 50
    for var in continous_vars: 
        print 'Variable: {} : Fires, then non-fires'.format(var)
        print '-' * 50
        for idx, df in enumerate(dfs): 
            f, axes = plt.subplots(1, 8, figsize=(20, 5))
            fire_var = fires[idx][var]
            non_fire_var = non_fires[idx][var]
            print fire_var.mean(), non_fire_var.mean()
            plot_var_dist(fire_var, categorical=False, ax=axes[0:4], show=False)
            plot_var_dist(non_fire_var, categorical=False, ax=axes[4:], show=False)
            plt.show()
    print 'Categorical Vars'
    print '-' * 50
    for var in categorical_vars: 
        print 'Variable: {} : Fires, then non-fires'.format(var)
        print '-' * 50
        for idx, df in enumerate(dfs):
            f, axes = plt.subplots(1, 2)
            plot_var_dist(fires[idx][var], categorical=True, ax=axes[0], show=False)
            plot_var_dist(non_fires[idx][var], categorical=True, ax=axes[1], show=False)
            plt.show()

check_dists(2012, continous_vars, categorical_vars)

check_dists(2013, continous_vars, categorical_vars)

check_dists(2014, continous_vars, categorical_vars)

check_dists(2015, continous_vars, categorical_vars)



