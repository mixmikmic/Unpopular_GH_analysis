from multiprocessing import Pool #witness the power
import wikipedia
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
get_ipython().magic('matplotlib inline')
from fuzzywuzzy import fuzz
from collections import defaultdict
from helper_functions import *

def extract_rotton_info_v2(webpage):
    
    master_dict = {}
    movie_rank_index = 0
    tomato_rating_index = 1
    movie_url_index = 2
    genre_name = webpage.split("https://www.rottentomatoes.com/top/bestofrt/top_100_")[1].strip("/")


    print("-------------","Processing: ",webpage,"---------------")

    soup = BeautifulSoup(requests.get(webpage).text,'lxml')

    top_100_of_sub_genre = soup.find_all(class_='table')[0].find_all('td')

    for _ in range(1,(int(len(top_100_of_sub_genre)/4)+1)):

        rank = top_100_of_sub_genre[movie_rank_index].text.strip()

        tomato_percentage = top_100_of_sub_genre[tomato_rating_index].find(class_='tMeterScore').text.strip()

        movie_name = top_100_of_sub_genre[movie_url_index].text.strip()
        movie_name = movie_name+" (film)"

        movie_url = base_url+top_100_of_sub_genre[movie_url_index].find('a').get('href')
        
        movie_page = BeautifulSoup(requests.get(movie_url).text, 'lxml')

        #audience rating is out of 5
        audience_rating = movie_page.find(class_="audience-info hidden-xs superPageFontColor").text.split()[2]
        rotton_info_extraction = movie_page.find("div", {"id": "scoreStats"}).text.split()
        
        rotton_average_rating = rotton_info_extraction[2].split('/')[0] #out of 10
        rotton_reviews_counted = rotton_info_extraction[5]
        
        if movie_name not in master_dict: #want to avoid duplicate movies across lists.
            master_dict[movie_name] = [rank, rotton_average_rating, rotton_reviews_counted, tomato_percentage, audience_rating]
            
        
        movie_rank_index +=4
        tomato_rating_index += 4
        movie_url_index += 4
        
    return master_dict

# def extract_movie_names(array):
#     movie_names = []

#     for index, val in enumerate(array):
#         genre_list = array[index][list(array[index].keys())[0]]
#         for row in genre_list:
#             clean = row[0].split('(')
#             name = clean[0].strip()
#             year = clean[1].strip(')')
#             movie_names.append((name, year))
    
#     return movie_names

# def extract_movie_names(array): #movie names will now be the key
#     movie_names = []

#     for index, val in enumerate(array):
#         genre_list = array[index][list(array[index].keys())[0]]
        
#         for row in genre_list:
#             movie_names.append(row[0])
    
#     return movie_names

genre_urls_to_scrape = extract_sub_genre_links(starting_url)

all_rotton_data = witness_the_power(extract_rotton_info_v2, genre_urls_to_scrape)

movie_database = extract_unique_movies_across_genre(all_rotton_data)

len(movie_database.keys())

pickle_object(all_rotton_data,"all_rotton_data")

pickle_object(movie_database,"movie_database")

list(movie_database.keys())

v = wikipedia.page(all_movie_names[0][0])

v

for i in dir(v)[39:]:
    print(i)

soup = BeautifulSoup(v.html(), 'lxml')

wikipedia_api_info = soup.find("table",{"class":"infobox vevent"})

result = {}
for tr in wikipedia_api_info.find_all('tr'):
    if tr.find('th'):
        result[tr.find('th').text] = tr.find('td')

result.keys()

result['Directed by'].text.strip()

result['Release date'].li.text.split("\xa0")[1]

result['Running time'].text.strip().split(" minutes")[0]

result['Box office'].text.strip().split('[')[0]

result['Budget'].text.strip().split("[")[0]

result['Language'].text.strip()

wikipedia_api_info.strip().split("\n") # very messy - lets trip the WIP tools!

import wptools
x = wptools.page(all_movie_names[0][0]).get() #got the information for mad max

x.wikidata #returns a nice dict of stuff that is also in the infobox.
#should use this to extract director name and date

director = x.wikidata['director']
director

month_released = x.wikidata['pubdate']
datetime.strptime(month_released[0].strip('+').split('T')[0], "%Y-%m-%d").month

soup_new = BeautifulSoup(x.wikitext, 'lxml')

soup_new.find('table', {"class":"infobox vevent"})

x.infobox

d = x.infobox

for k,v in d.items():
    print(k,v)
    print()

h = d['released'].strip('{').strip('}').strip('Film date|').split("|")

h

for index, value in enumerate(h):
    if value == 'United States':
        month = h[index-2]
        print(month)



