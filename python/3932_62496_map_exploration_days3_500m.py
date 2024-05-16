import pandas as pd
from dsfuncs.geo_plotting import USMapBuilder
import matplotlib.pyplot as plt
import fiona
import subprocess
import datetime
get_ipython().magic('matplotlib inline')

def read_df(year, modis=True): 
    """This function will read in a year of data, and add a month column. 
    
    Args: 
        year: str
        modis: bool
            Whether to use the modis or viirs data for plotting. 
        
    Return:
        Pandas DataFrame
    """
    if modis: 
        output_df = pd.read_csv('../../../data/csvs/day3_500m/detected_fires_MODIS_' + str(year) + '.csv', 
                                parse_dates=['date'], true_values=['t'], false_values=['f'])
    else: 
         output_df = pd.read_csv('../../../data/csvs/day3_500m/detected_fires_VIIRS_' + str(year) + '.csv', 
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

def read_n_parse(year, state_names, county_names=None, months=None, plotting=False, dt=None): 
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
    
    fires_df = read_df(year)
    if state_names: 
        fires_df = grab_by_location(fires_df, state_names, county_names)
    
    if months or dt: 
        fires_df = grab_by_date(fires_df, months, dt)
    
    if plotting: 
        fires_df = format_df(fires_df)
    return fires_df

def plot_states(year, state_names, months=None, plotting=True): 
    """Plot a state map and the given fires data points for that state. 
    
    Args: 
        year: str
        state_names: set (or other iterable) of state names
            State names to grab for plotting. 
        months: set (or other iterable) of month names 
            Month names to grab for plotting. 
        plotting: bool 
            Whether or not to format the data for plotting. 
    
    Return: Plotted Basemap
    """
    ax = plt.subplot(1, 2, 1)
    state_map = USMapBuilder('data/state_shapefiles_2014/cb_2014_us_state_500k2.shp', geo_level='State', 
                        state_names=state_names, ax=ax, border_padding=1)
    fires_data = read_n_parse(year, state_names, months=months, plotting=plotting)
    fires_data_trues = fires_data[fires_data[:,2] == 'ro']
    fires_data_falses = fires_data[fires_data[:,2] == 'go']
    print fires_data_trues.shape, fires_data_falses.shape
    state_map.plot_points(fires_data_trues)
    ax = plt.subplot(1, 2, 2)
    state_map = USMapBuilder('data/state_shapefiles_2014/cb_2014_us_state_500k2.shp', geo_level='State',  
                        state_names=state_names, ax=ax, border_padding=1)
    state_map.plot_points(fires_data_falses)
    plt.show()

years = ['2012', '2013', '2014', '2015']
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 
         'November', 'December']

state_names = ['California']
for month in months: 
    for year in years: 
        print 'Year: {}, Month: {}'.format(year, month)
        plot_states(year, state_names, months=[month])

state_names = ['Colorado']
for month in months: 
    for year in years: 
        print 'Year: {}, Month: {}'.format(year, month)
        plot_states(year, state_names, months=[month])

state_names = ['Montana']
for month in months: 
    for year in years: 
        print 'Year: {}, Month: {}'.format(year, month)
        plot_states(year, state_names, months=[month])

state_names = ['Washington']
for month in months: 
    for year in years: 
        print 'Year: {}, Month: {}'.format(year, month)
        plot_states(year, state_names, months=[month])

state_names = ['Texas']
for month in months: 
    for year in years: 
        print 'Year: {}, Month: {}'.format(year, month)
        plot_states(year, state_names, months=[month])

state_names = ['California']
for year in years: 
    print 'Year: {}'.format(year)
    plot_states(year, state_names)

state_names = ['Colorado']
for year in years: 
    print 'Year: {}'.format(year)
    plot_states(year, state_names)

state_names = ['Montana']
for year in years: 
    print 'Year: {}'.format(year)
    plot_states(year, state_names)

state_names = ['Washington']
for year in years: 
    print 'Year: {}'.format(year)
    plot_states(year, state_names)

state_names = ['Texas']
for year in years: 
    print 'Year: {}'.format(year)
    plot_states(year, state_names)

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

def plot_st_fires_boundaries(dt, st_name, st_abbrev, st_fips): 
    """Plot the fire boundaries for a year and given state.
    
    Args
    ----
        dt: str
            Contains date of boundaries to grab. 
        state_name: str
            Holds state to plot. 
        st_abbrev: str
            Holds the states two-letter abbreviation. 
        st_fips: str
            Holds the state's fips number. 
        
    Return: Plotted Basemap
    """
    
    boundaries_filepath = grab_fire_perimeters(dt, st_name, st_abbrev, st_fips)
    st_map = USMapBuilder('data/state_shapefiles_2014/cb_2014_us_state_500k2.shp', geo_level='State', 
                          state_names=[st_name], border_padding=1)
    st_map.plot_boundary(boundaries_filepath)
    plt.show()
    
def plot_counties_fires_boundaries(year, state_name, st_abbrev, st_fips, county_name, 
                                   months=None, plotting=True, markersize=4): 
    """Plot a county map in a given state, including any fire perimeter boundaries and potentially
    detected fires in those counties. 
    
    Args: 
        year: str
        state_name: str
            State name to grab for plotting. 
        st_abbrev: str
            Holds the states two-letter abbrevation
        st_fips: str
            Holds the state's fips number. 
        county_name: str or iterable of strings 
            County names to grab for plotting. 
        months: set (or other iterable) of strings (optional)
            Month names to grab for plotting
        plotting: bool
            Whether or not to format the data for plotting. 
    """
    
    county_name = county_name if isinstance(county_name, list) else [county_name]
    county_map = USMapBuilder('data/county_shapefiles_2014/cb_2014.shp', geo_level='County', 
                state_names=[state_name], county_names=county_name, figsize=(40, 20), border_padding=0.1)
    boundaries_filepath = grab_fire_perimeters(year, state_name, st_abbrev, st_fips)
    county_map.plot_boundary(boundaries_filepath)
    if plotting: 
        fires_data = read_n_parse(year, state_name, months=months, plotting=plotting)
        county_map.plot_points(fires_data, markersize)
    plt.show()

def plot_county_dt(st_name, st_abbrev, st_fips, county_name, fires_dt=None, perims_dt=None, 
                  markersize=4): 
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
        markersize (optional): int
            Used to control the size of marker to use for plotting fire points. 
    """
    
    year = fires_dt.split('-')[0]
    perims_dt = fires_dt if not perims_dt else perims_dt
    county_names = [county_name] if not isinstance(county_name, list) else county_name
    fires_data = read_n_parse(year, state_names=st_name, county_names=county_name, dt=fires_dt, plotting=True)
    county_map = USMapBuilder('data/county_shapefiles_2014/cb_2014.shp', geo_level='County', 
                state_names=[st_name], county_names=county_names, figsize=(40, 20), border_padding=0.1)
    fire_boundaries = grab_fire_perimeters(perims_dt, st_name, st_abbrev, st_fips)
    try: 
        county_map.plot_boundary(fire_boundaries)
    except Exception as e:
        print e
    print len(fires_data)
    county_map.plot_points(fires_data, markersize=markersize)
    plt.show()

for dt in xrange(1, 31): 
    dt = '2015-06-0' + str(dt) if dt < 10 else '2015-06-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Colorado', 'co', '08', 'Moffat', dt)

for dt in xrange(1, 31): 
    dt = '2012-06-0' + str(dt) if dt < 10 else '2012-06-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Colorado', 'co', '08', 'Mesa', dt)

for dt in xrange(1, 31): 
    dt = '2013-07-0' + str(dt) if dt < 10 else '2013-07-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Colorado', 'co', '08', ['Hinsdale', 'Mineral'], dt)

for dt in xrange(1, 31): 
    dt = '2015-07-0' + str(dt) if dt < 10 else '2015-07-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Colorado', 'co', '08', 'Logan', dt)

for dt in xrange(1, 31): 
    dt = '2014-08-0' + str(dt) if dt < 10 else '2014-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Colorado', 'co', '08', 'Las Animas', dt)

for dt in xrange(1, 31): 
    dt = '2014-01-0' + str(dt) if dt < 10 else '2014-01-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', 'Los Angeles', dt)

for dt in xrange(1, 29): 
    dt = '2014-02-0' + str(dt) if dt < 10 else '2014-02-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', ['Glenn'], dt)

for dt in xrange(1, 31): 
    dt = '2013-05-0' + str(dt) if dt < 10 else '2013-05-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', ['Ventura'], dt, markersize=2)

for dt in xrange(1, 31): 
    dt = '2012-07-0' + str(dt) if dt < 10 else '2012-07-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', ['Fresno'], dt, markersize=4)

for dt in xrange(1, 31): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('California', 'ca', '06', ['Mono'], dt, markersize=4)

for dt in xrange(1, 31): 
    dt = '2012-07-0' + str(dt) if dt < 10 else '2012-07-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Montana', 'mt', '30', ['Big Horn'], dt, markersize=4)

for dt in xrange(1, 31): 
    dt = '2013-08-0' + str(dt) if dt < 10 else '2013-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Montana', 'mt', '30', ['Ravalli'], dt, markersize=4)

for dt in xrange(1, 31): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Montana', 'mt', '30', ['Lewis and Clark'], dt, markersize=4)

for dt in xrange(1, 31): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Montana', 'mt', '30', ['Pondera'], dt, markersize=4)

for dt in xrange(1, 31): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Montana', 'mt', '30', ['Lake', 'Missoula', 'Ravalli', 'Mineral', 'Sanders'], dt, markersize=4)

for dt in xrange(1, 31): 
    dt = '2015-08-0' + str(dt) if dt < 10 else '2015-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Washington', 'wa', '53', ['Columbia', 'Garfield'], dt, markersize=4)

for dt in xrange(1, 31): 
    dt = '2012-08-0' + str(dt) if dt < 10 else '2012-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Washington', 'wa', '53', ['Asotin'], dt, markersize=4)

for dt in xrange(1, 31): 
    dt = '2014-03-0' + str(dt) if dt < 10 else '2014-03-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Washington', 'wa', '53', ['Walla Walla', 'Franklin', 'Benton'], dt, markersize=4)

for dt in xrange(1, 31): 
    dt = '2013-07-0' + str(dt) if dt < 10 else '2013-07-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Washington', 'wa', '53', ['Klickitat'], dt, markersize=4)

for dt in xrange(1, 31): 
    dt = '2014-08-0' + str(dt) if dt < 10 else '2014-08-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Washington', 'wa', '53', ['Mason'], dt, markersize=4)

for dt in xrange(1, 31): 
    dt = '2012-01-0' + str(dt) if dt < 10 else '2012-01-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Texas', 'tx', '48', ['Cameron', 'Hidalgo', 'Willacy'], dt, markersize=4)

for dt in xrange(1, 31): 
    dt = '2014-04-0' + str(dt) if dt < 10 else '2014-04-' + str(dt)
    print 'Date: {}'.format(dt)
    print '-' * 50
    plot_county_dt('Texas', 'tx', '48', ['Presidio', 'Brewster', 'Jeff Davis'], dt, markersize=4)



