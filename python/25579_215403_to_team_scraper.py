## Building out scraper for nfl team stats
import pandas as pd
import urllib
from bs4 import BeautifulSoup
import time


# Generating a range of ints by 100 to append to url at the end
# to navigate between pages
leaf = range(0, 9100, 100)

## root url below:
base_url = ('http://www.pro-football-reference.com/play-index/tgl_finder.cgi?request=1&match=game&year_min=1999&year_max=2016&game_type=&game_num_min=0&game_num_max=99&week_num_min=0&week_num_max=99&game_day_of_week=&game_time=&time_zone=&surface=&roof=&temperature=&temperature_gtlt=lt&game_location=&game_result=&overtime=&league_id=NFL&team_id=&opp_id=&team_div_id=&opp_div_id=&team_conf_id=&opp_conf_id=&date_from=&date_to=&team_off_scheme=&opp_off_scheme=&team_def_align=&opp_def_align=&stadium_id=&c1stat=points&c1comp=gt&c1val=&c2stat=tot_yds&c2comp=gt&c2val=&c3stat=pass_cmp_opp&c3comp=gt&c3val=&c4stat=rush_att_opp&c4comp=gt&c4val=&c5comp=&c5gtlt=lt&c6mult=1.0&c6comp=&order_by=game_date&order_by_asc=&matching_games=1&conference_game=&division_game=&tm_is_playoff=&opp_is_playoff=&tm_is_winning=&opp_is_winning=&tm_scored_first=&tm_led=&tm_trailed=&tm_won_toss=&offset=')

## this will loop through the leaf list and creat my entire list of urls
## and append to the empty list "url_list"
url_list = []
for i in leaf:
    url_list.append(base_url + str(i))

## I am now creating a test link list so that I can work through the code
## and debug while working with a less extensive dataset

test_urls = url_list[0:1]

## as you can see below, my test_urls list is composed of only first link
print test_urls

headers = []

html = urllib.urlopen(test_urls[0])
soup = BeautifulSoup(html, 'html.parser')

tab_header = soup.findAll('th')
for i in tab_header:
    headers.append(i.renderContents())
print len(headers)

## because of some extra headers mixed in, the actual column headers
## that we want are interspersed in the following range
print headers[6:42]

## these names are a bit confusing due to the organization in the table
## so they will probably just be manually recorded later on

## The below code is an effort to scrape the stats for each game since 1999

## I am manually recording a list of column names and changing them to
## more easily interpretable names
team_data_cols = ['rk', 'team', 'year', 'date', 'east_time', 'loc_time', 'blank_@',
                 'opp', 'week', 'team_game#', 'day', 'result', 'ot', 'pf', 'pa', 'pdiff',
                 'pcomb', 'tot_yds', 'off_plys', 'off_yds/ply', 'def_plys', 'def_yds/ply', 'to_lost',
                  'off_time_poss','game_duration', 'opp_completions', 'opp_pass_att', 'opp_comp_perc',
                 'opp_pass_yds', 'opp_pass_tds', 'opp_int_thrown', 'opp_sacks_taken', 'opp_sacks_yds_lost',
                 'opp_rush_atts', 'opp_rush_yds', 'opp_rush_yds/att', 'opp_rush_td']

## redefining our html and soup just to make sure I am referencing the
## correct link and soup
html = urllib.urlopen(test_urls[0])
soup = BeautifulSoup(html, 'html.parser')

## creating an empty list to append the data to
data_points = []

## creating variable that consists of the body of webpage we are interested
## in combing through
body = soup.findAll('tbody')

## this variable creates a mass of the individual rows of the data to
## work our way through
indiv_rows = body[0].findAll('td')

## i will loop through each row to strip out the data
for row in indiv_rows:
    # the line below redefines my soup as the contents of each individual row
    inner_soup = BeautifulSoup(row.renderContents(), 'html.parser')
    # this adds the data to the empty list
    # the .text function strips hyperlinks and returns the text value only
    data_points.append(inner_soup.text)

## since data points is just one long list, i need to break it up into
## individual chunks so it can be added to our dataframe
chunks = [data_points[x:x+37] for x in range(0, len(data_points), 37)]

## Here I create a dataframe from the newly formed chunks and have
## the newly defined columns as my column names
team_df = pd.DataFrame(chunks, columns = team_data_cols)
team_df

### I am going to consolidate all the above code and unleash on all
### of the webpages we are going for in order to scrape all the data

## i am also importing time to put a sleep time in my loop
import time

leaf = range(0, 9100, 100)

base_url = ('http://www.pro-football-reference.com/play-index/tgl_finder.cgi?request=1&match=game&year_min=1999&year_max=2016&game_type=&game_num_min=0&game_num_max=99&week_num_min=0&week_num_max=99&game_day_of_week=&game_time=&time_zone=&surface=&roof=&temperature=&temperature_gtlt=lt&game_location=&game_result=&overtime=&league_id=NFL&team_id=&opp_id=&team_div_id=&opp_div_id=&team_conf_id=&opp_conf_id=&date_from=&date_to=&team_off_scheme=&opp_off_scheme=&team_def_align=&opp_def_align=&stadium_id=&c1stat=points&c1comp=gt&c1val=&c2stat=tot_yds&c2comp=gt&c2val=&c3stat=pass_cmp_opp&c3comp=gt&c3val=&c4stat=rush_att_opp&c4comp=gt&c4val=&c5comp=&c5gtlt=lt&c6mult=1.0&c6comp=&order_by=game_date&order_by_asc=&matching_games=1&conference_game=&division_game=&tm_is_playoff=&opp_is_playoff=&tm_is_winning=&opp_is_winning=&tm_scored_first=&tm_led=&tm_trailed=&tm_won_toss=&offset=')

url_list = []
for i in leaf:
    url_list.append(base_url + str(i))
    
team_data_cols = ['rk', 'team', 'year', 'date', 'east_time', 'loc_time', 'blank_@',
                 'opp', 'week', 'team_game#', 'day', 'result', 'ot', 'pf', 'pa', 'pdiff',
                 'pcomb', 'tot_yds', 'off_plys', 'off_yds/ply', 'def_plys', 'def_yds/ply', 'to_lost',
                  'off_time_poss','game_duration', 'opp_completions', 'opp_pass_att', 'opp_comp_perc',
                 'opp_pass_yds', 'opp_pass_tds', 'opp_int_thrown', 'opp_sacks_taken', 'opp_sacks_yds_lost',
                 'opp_rush_atts', 'opp_rush_yds', 'opp_rush_yds/att', 'opp_rush_td']

data_points = []

## this initial loop has been added to circulate through all the links collected
# also adding count to check my status while conducting the loop
count = 0
for i in url_list:
    html = urllib.urlopen(i)
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.findAll('tbody')
    indiv_rows = body[0].findAll('td')
    for row in indiv_rows:
        inner_soup = BeautifulSoup(row.renderContents(), 'html.parser')
        data_points.append(inner_soup.text)
    ## adding a sleep time to not ping website too frequently
    count += 1
    print "you have completed this many loops:  %d" % count
    #time.sleep(1)

chunks = [data_points[x:x+37] for x in range(0, len(data_points), 37)]

team_df = pd.DataFrame(chunks, columns = team_data_cols)
team_df

print team_df.columns
team_df.head()

## this will drop the first col which is a repeat of the index
team_df = team_df.drop('Unnamed: 0', axis = 1)

## this code will mark every game with @ in the column as an away game
team_df['blank_@'] = team_df['blank_@'].map({'@':'away'})

## this will mark all the na's as home games
team_df['blank_@'].fillna(value = 'home', inplace=True)

## this renames the column as the game location
team_df.rename(columns = {'blank_@':'location'}, inplace = True)

team_df.head()

## checking to see where null values are to fix
team_df.isnull().sum()

print team_df.ot.value_counts()
print
## i am going to change these to zero for no ot and 1 for ot
team_df['ot'] = team_df['ot'].fillna(0)

## checking to confirm it worked correctly
print team_df.ot.value_counts()

## replacding the OT values with a 1
team_df.ot.replace('OT', 1, inplace = True)

team_df.ot.value_counts()

## finding the remainder of the null values
team_df.isnull().sum()

## checking value counts to confirm my thought that the nulls are zeros here
print team_df.to_lost.value_counts()

## going to fill na's with the value zero
team_df.to_lost.fillna(0, inplace = True)

## checking to confirm that all 1851 nulls are now zeros
print team_df.to_lost.value_counts()

## checking datatypes to see if there is anything that currently needs to be changed
team_df.dtypes

## I want to create a win/loss column by itself
## creating an empty column to append data to
#team_df['win_loss'] = team_df.result

# creating an empty list and appending the first character of each row to it
wl_list = []
wl_list
for i in team_df.result:
    wl_list.append(i[0])
wl_list = pd.Series(wl_list)


## I do the first character only because it is either a L, W, or T for tie

## this creates a new column (win_loss) and sets it equal to the wl_list series
team_df['win_loss'] = wl_list
team_df.head()

## this is a check to confirm that the correct changes were made
team_df.win_loss.value_counts()

## this shows the rows where the game ended in a tie
team_df[team_df.win_loss.str.contains('T')]

print team_df.columns

## I am creating a csv file from the newly formed team_df and
## exporting to to my current working directory
team_df.to_csv('team_data_df', encoding = 'utf-8')

## I will now be creating a database in postgres in order to add
## this dataframe as a table to perform queries on outside of python

from sqlalchemy import create_engine
import psycopg2

engine = create_engine('postgresql://TerryONeill@localhost:5432/nfl_capstone')

## this is adding the dataframe to my newly created database in psql as
## a table named 'team_data_table'
team_df.to_sql('team_data_table', engine)




