from multiprocessing import Pool #witness the power
import wikipedia
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import re
import time
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from datetime import datetime
from helper_functions import *
get_ipython().magic('matplotlib inline')

movie_db = unpickle_object("movie_database.pkl")

len(sorted(movie_db.keys()))

def extract_wiki_infobox():
    
    regex = r" *\[[^\]]*.*"
    regex2 = r" *\([^\)].*"
    regex3 = r" *\/[^\)]*.*"
    regex4 = r" *\,[^\)].*"
    regex5 = r".*(?=\$)"
    regex6 = r".*(?=\£)"
    regex7 = r"\–.*$"
    regex_date = r"^[^\(]*"
    regex_date_2 = r" *\)[^\)].*"
    subset=''



    for key in sorted(movie_db.keys()):
        if len(movie_db[key]) == 6:
            html_url = movie_db[key][-1].url
            info_box_dictionary = {}
            soup = BeautifulSoup(movie_db[key][5].html(), 'lxml')
            wikipedia_api_info = soup.find("table",{"class":"infobox vevent"})

            info_box_dictionary = {}

            for tr in wikipedia_api_info.find_all('tr'):
                if tr.find('th'):
                    info_box_dictionary[tr.find('th').text] = tr.find('td')

            try: #done
                date = info_box_dictionary['Release date'].text
                date = re.sub(regex_date, subset, date)
                try:
                    date = date.split()[0].strip("(").strip(")")
                    date = re.sub(regex_date_2,subset, date)
                except IndexError:
                    date = info_box_dictionary['Release date'].text
                    date = re.sub(regex_date, subset, date)
            except KeyError:
                date = np.nan

            try: #done
                runtime = info_box_dictionary['Running time'].text
                runtime = re.sub(regex, subset, runtime)
                runtime = re.sub(regex2, subset, runtime)
            except KeyError:
                runtime = np.nan

            try: #done
                boxoffice = info_box_dictionary['Box office'].text
                boxoffice = re.sub(regex, subset, boxoffice)
                boxoffice = re.sub(regex6, subset, boxoffice)
                boxoffice = re.sub(regex5, subset, boxoffice)
                if "billion" not in boxoffice:
                    boxoffice = re.sub(regex7, subset, boxoffice)
                    boxoffice = re.sub(regex2, subset, boxoffice)
            except KeyError:
                boxoffice = np.nan

            try:#done
                budget = info_box_dictionary['Budget'].text
                budget = re.sub(regex, subset, budget)
                budget = re.sub(regex7, subset, budget)
                if "$" in budget:
                    budget = re.sub(regex5, subset, budget)
                    budget = re.sub(regex2, subset, budget)
                if "£" in budget:
                    budget = re.sub(regex6, subset, budget)
                    budget = re.sub(regex2, subset, budget)
                budget = re.sub(regex5, subset, budget)
            except KeyError:
                budget = np.nan

            try:#done
                country = info_box_dictionary['Country'].text.strip().lower()
                country = re.sub(regex, subset, country) #cleans out a lot of gunk
                country = re.sub(regex2, subset, country)
                country = re.sub(regex3, subset, country)
                country = re.sub(regex4, subset, country)
                country = country.split()
                if country[0] == "united" and country[1] == "states":
                    country = country[0]+" "+country[1]
                elif country[0] =="united" and country[1] == "kingdom":
                    country = country[0] +" "+ country[1]
                else:
                    country = country[0]
            except KeyError:
                country = np.nan

            try:#done
                language = info_box_dictionary['Language'].text.strip().split()[0]
                language = re.sub(regex, subset, language)
            except KeyError:
                language = np.nan

            movie_db[key].append(date)
            movie_db[key].append(runtime)
            movie_db[key].append(boxoffice)
            movie_db[key].append(budget)
            movie_db[key].append(country)
            movie_db[key].append(language)

        

extract_wiki_infobox()















