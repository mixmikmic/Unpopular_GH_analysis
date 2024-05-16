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
from datetime import datetime
from helper_functions import *
get_ipython().magic('matplotlib inline')

# def extract_wiki_info(lst): No need to ever run this function again
#     success_list = []
#     error_list = []
#     add_to_success = []
#     potential_renaming = {}
#     regex = r"\([*(\d+)[^\d]*\)" #removes (year_value)
#     subset=""
    
#     for movie_title in lst:
#         try:
#             wiki_html = wikipedia.page(movie_title)
#             success_list.append((movie_title,wiki_html))
#         except wikipedia.exceptions.DisambiguationError as e:
#             potential_renaming[movie_title] = e.options
#             continue
#         except wikipedia.exceptions.PageError as e2:
#             try:
#                 clean_movie_title = re.sub(regex,subset, movie_title) #removes (year_digits) from movie name
#                 clean_wiki_html = wikipedia.page(clean_movie_title)
#                 add_to_success.append((movie_title,clean_movie_title,clean_wiki_html))
#             except:
#                 error_list.append(movie_title)
#                 continue
#         except:
#             error_list.append(movie_title)
#             continue
#     return success_list, add_to_success, error_list, potential_renaming

# s_list, add_success, e_list, rename_dict = extract_wiki_info(movie_title_list)

len(s_list)+len(add_success)+len(e_list)+len(rename_dict) #we have the same length as final_list! All movies processed

pickle_object(s_list, "s_list")
pickle_object(add_success, "add_success")
pickle_object(e_list, "e_list")
pickle_object(rename_dict, "rename_dict")

# for i in s_list: worked perfectly
#     movie_dictionary[i[0]].append(i[1])

# for i in add_success: worked perfectly
#     movie_dictionary[i[1]] = movie_dictionary[i[0]]
#     movie_dictionary[i[1]].append(i[2])
#     del movie_dictionary[i[0]]

# for i in e_list: #placeholder to maintain dimensions. the movies in e_list dont have wikipedia_objects
#     movie_dictionary[i].append("")

# had to hard code these movies that had errors
# movie_dictionary[e_list[0]].append("2014-05-15")
# movie_dictionary[e_list[0]].append(96.0)
# movie_dictionary[e_list[0]].append('$10 million')
# movie_dictionary[e_list[0]].append(np.nan)
# movie_dictionary[e_list[0]].append("Arabic")
# movie_dictionary[e_list[0]].append("France")

# movie_dictionary[e_list[1]].append("2015-09-12")
# movie_dictionary[e_list[1]].append(109.0)
# movie_dictionary[e_list[1]].append("$8.7 million")
# movie_dictionary[e_list[1]].append(np.nan)
# movie_dictionary[e_list[1]].append("Spanish")
# movie_dictionary[e_list[1]].append("Spain")

# movie_dictionary[e_list[2]].append("2014-09-25")
# movie_dictionary[e_list[2]].append(98.0)
# movie_dictionary[e_list[2]].append("$5.6 million")
# movie_dictionary[e_list[2]].append(np.nan)
# movie_dictionary[e_list[2]].append("German")
# movie_dictionary[e_list[2]].append("Germany")

# movie_dictionary[e_list[3]].append("1972-05-13")
# movie_dictionary[e_list[3]].append(166.0)
# movie_dictionary[e_list[3]].append(np.nan)
# movie_dictionary[e_list[3]].append("$0.829 million")
# movie_dictionary[e_list[3]].append("Russian")
# movie_dictionary[e_list[3]].append("Russia")

# movie_dictionary[e_list[4]].append("2016-05-12")
# movie_dictionary[e_list[4]].append(156.0)
# movie_dictionary[e_list[4]].append("$51.3 million")
# movie_dictionary[e_list[4]].append(np.nan)
# movie_dictionary[e_list[4]].append("South Korea")
# movie_dictionary[e_list[4]].append("Korean")

# movie_dictionary[e_list[5]].append("1979-08-17")
# movie_dictionary[e_list[5]].append(93.0)
# movie_dictionary[e_list[5]].append("$20 million") #box
# movie_dictionary[e_list[5]].append("$4 million") #budget
# movie_dictionary[e_list[5]].append("United Kingdom")
# movie_dictionary[e_list[5]].append("English")

# movie_dictionary[e_list[6]].append("2017-02-10")
# movie_dictionary[e_list[6]].append(79.0)
# movie_dictionary[e_list[6]].append("$2.6 million")
# movie_dictionary[e_list[6]].append("$1 million")
# movie_dictionary[e_list[6]].append("Turkey")
# movie_dictionary[e_list[6]].append("Turkish")

# movie_dictionary[e_list[7]].append("2014-07-17")
# movie_dictionary[e_list[7]].append(125.0)
# movie_dictionary[e_list[7]].append("$1.9 million")
# movie_dictionary[e_list[7]].append("$2.1 million")
# movie_dictionary[e_list[7]].append("New Zealand")
# movie_dictionary[e_list[7]].append("English")

# movie_dictionary[e_list[8]].append("2015-01-28")
# movie_dictionary[e_list[8]].append(82.0)
# movie_dictionary[e_list[8]].append(np.nan)
# movie_dictionary[e_list[8]].append(np.nan)
# movie_dictionary[e_list[8]].append("United States")
# movie_dictionary[e_list[8]].append("English")

# movie_dictionary[e_list[9]].append("2009-05-29")
# movie_dictionary[e_list[9]].append(99.0)
# movie_dictionary[e_list[9]].append("$90.8 million")
# movie_dictionary[e_list[9]].append("$30 million")
# movie_dictionary[e_list[9]].append("United States")
# movie_dictionary[e_list[9]].append("English")

# movie_dictionary[e_list[10]].append("2006-07-27")
# movie_dictionary[e_list[10]].append(119.0)
# movie_dictionary[e_list[10]].append("$89.4 million")
# movie_dictionary[e_list[10]].append("$11 million")
# movie_dictionary[e_list[10]].append("South Korea")
# movie_dictionary[e_list[10]].append("Korean")

# movie_dictionary[e_list[11]].append("2017-03-10")
# movie_dictionary[e_list[11]].append(99.0)
# movie_dictionary[e_list[11]].append("$0.515 million")
# movie_dictionary[e_list[11]].append("$3.8 million")
# movie_dictionary[e_list[11]].append("France")
# movie_dictionary[e_list[11]].append("French")

# movie_dictionary[e_list[12]].append("1993-12-03")
# movie_dictionary[e_list[12]].append(92.0)
# movie_dictionary[e_list[12]].append("$0.621 million")
# movie_dictionary[e_list[12]].append("$2 million")
# movie_dictionary[e_list[12]].append("Mexico")
# movie_dictionary[e_list[12]].append("Spanish")

# movie_dictionary[e_list[13]].append("2004-09-14")
# movie_dictionary[e_list[13]].append(98.0)
# movie_dictionary[e_list[13]].append("$7.5 million")
# movie_dictionary[e_list[13]].append(np.nan)
# movie_dictionary[e_list[13]].append("United Kingdom")
# movie_dictionary[e_list[13]].append("English")

# movie_dictionary[e_list[14]].append("1995-04-28")
# movie_dictionary[e_list[14]].append(120.0)
# movie_dictionary[e_list[14]].append("$3.1 million")
# movie_dictionary[e_list[14]].append(np.nan)
# movie_dictionary[e_list[14]].append("United States")
# movie_dictionary[e_list[14]].append("English")

# movie_dictionary[e_list[15]].append("1968-09-18")
# movie_dictionary[e_list[15]].append(149.0)
# movie_dictionary[e_list[15]].append("$58.5 million")
# movie_dictionary[e_list[15]].append("$14.1 million")
# movie_dictionary[e_list[15]].append("United States")
# movie_dictionary[e_list[15]].append("English")

# movie_dictionary[e_list[16]].append("1986-11-07")
# movie_dictionary[e_list[16]].append(114.0)
# movie_dictionary[e_list[16]].append("$2.8 million")
# movie_dictionary[e_list[16]].append("$4 million")
# movie_dictionary[e_list[16]].append("United Kingdom")
# movie_dictionary[e_list[16]].append("English")

# movie_dictionary[e_list[17]].append("1975-09-25")
# movie_dictionary[e_list[17]].append(88.0)
# movie_dictionary[e_list[17]].append(np.nan)
# movie_dictionary[e_list[17]].append(np.nan)
# movie_dictionary[e_list[17]].append("France")
# movie_dictionary[e_list[17]].append("French")

# movie_dictionary[e_list[18]].append("1951-06-30")
# movie_dictionary[e_list[18]].append(101.0)
# movie_dictionary[e_list[18]].append("$7 million")
# movie_dictionary[e_list[18]].append("$1.2 million")
# movie_dictionary[e_list[18]].append("United States")
# movie_dictionary[e_list[18]].append("English")

# movie_dictionary[e_list[19]].append("2016-04-08")
# movie_dictionary[e_list[19]].append(102.0)
# movie_dictionary[e_list[19]].append("$34.6 million")
# movie_dictionary[e_list[19]].append("$13 million")
# movie_dictionary[e_list[19]].append("United Kingdom")
# movie_dictionary[e_list[19]].append("English")

# movie_dictionary[e_list[20]].append("2002-09-06")
# movie_dictionary[e_list[20]].append(113.0)
# movie_dictionary[e_list[20]].append(np.nan)
# movie_dictionary[e_list[20]].append(np.nan)
# movie_dictionary[e_list[20]].append("Denmark")
# movie_dictionary[e_list[20]].append("Danish")

# movie_dictionary[e_list[21]].append("2013-08-16")
# movie_dictionary[e_list[21]].append(82.0)
# movie_dictionary[e_list[21]].append("$0.199 million")
# movie_dictionary[e_list[21]].append(np.nan)
# movie_dictionary[e_list[21]].append("United States")
# movie_dictionary[e_list[21]].append("English")

# movie_dictionary[e_list[22]].append("2014-10-15")
# movie_dictionary[e_list[22]].append(110)
# movie_dictionary[e_list[22]].append("$3.6 million")
# movie_dictionary[e_list[22]].append(np.nan)
# movie_dictionary[e_list[22]].append("France")
# movie_dictionary[e_list[22]].append("French")

# movie_dictionary[e_list[23]].append("2006-05-05")
# movie_dictionary[e_list[23]].append(125.0)
# movie_dictionary[e_list[23]].append("$0.568 million")
# movie_dictionary[e_list[23]].append(np.nan)
# movie_dictionary[e_list[23]].append("United States")
# movie_dictionary[e_list[23]].append("English")

correct_rename = {"3:10 to Yuma (2007) (film)": '3:10 to Yuma (2007 film)', 'About a Boy (2002) (film)':"About a Boy (film)",'Akira (1988) (film)':'Akira (1988 film)',
                 'Aladdin (1992) (film)':"Aladdin (1992 Disney film)", 'All Quiet on the Western Front (1930) (film)': 'All Quiet on the Western Front (1930 film)',
                 'Altered States (1980) (film)':"Altered States", 'Bamboozled (2000) (film)':"Bamboozled", 'Bridge of Spies (2015) (film)': 'Bridge of Spies (film)',
                 'Broadcast News (1987) (film)':'Broadcast News (film)','City Lights (1931) (film)': 'City Lights (1931 Film)','Dracula (1931) (film)': 'Dracula (1931 English-language film)',
                  'E.T. The Extra-Terrestrial (1982) (film)':"E.T. the Extra-Terrestrial", 'Enough Said (2013) (film)':'Enough Said (film)',
                 'Fantasia (1940) (film)':'Fantasia (1940 film)', 'From Here to Eternity (1953) (film)':"From Here to Eternity",
                 'Gentlemen Prefer Blondes (1953) (film)':'Gentlemen Prefer Blondes (1953 film)','Get Out (2017) (film)':'Get Out (film)',
                 'Hairspray (1988) (film)': 'Hairspray (1988 film)','Hedwig and the Angry Inch (2001) (film)': 'Hedwig and the Angry Inch (film)',
                 'Hell or High Water (2016) (film)': 'Hell or High Water (film)', "I'll See You in My Dreams (2015) (film)":"I'll See You in My Dreams (2015 film)",
                 "I'm Still Here (2010) (film)":"I'm Still Here (2010 film)", 'In the Heat of the Night (1967) (film)': 'In the Heat of the Night (film)','Inside Job (2010) (film)':'Inside Job (2010 film)',
                 'Invincible (2006) (film)':'Invincible (2006 film)','Last Train Home (2010) (film)':'Last Train Home (film)', 'On the Waterfront (1954) (film)':'On the Waterfront',
                 'Once Upon a Time in the West (1968) (film)':'Once Upon a Time (1918 film)',"One Flew Over the Cuckoo's Nest (1975) (film)":"One Flew Over the Cuckoo's Nest (film)",
                 'Only Yesterday (2016) (film)':"Only Yesterday (1991 film)",'Pina (2011) (film)': 'Pina (film)', 'Red Hill (2010) (film)':'Red Hill (film)',
                 "Rosemary's Baby (1968) (film)":"Rosemary's Baby (film)",'Spring (2015) (film)':'Spring (2014 film)','Texas Rangers (2001) (film)':"Texas Rangers (film)",
                 'The 39 Steps (1935) (film)': 'The 39 Steps (1935 film)','The Claim (2000) (film)':"The Claim", 'The Commitments (1991) (film)': 'The Commitments (film)',
                 'The Dead Zone (1983) (film)':'The Dead Zone (film)', 'The French Connection (1971) (film)': "The French Connection (film)",
                 'The Good, the Bad and the Ugly (1966) (film)':"The Good, the Bad and the Ugly",'The Grapes of Wrath (1940) (film)':'The Grapes of Wrath (film)',
                 'The Horse Whisperer (1998) (film)':'The Horse Whisperer (film)', 'The Innocents (1961) (film)': 'The Innocents (1961 film)',
                 'The Leopard (1963) (film)':'The Leopard (1963 film)','The Manchurian Candidate (1962) (film)': 'The Manchurian Candidate (1962 film)',
                 'The Missing (2003) (film)':'The Missing (2003 film)','The Night of the Hunter (1955) (film)':'The Night of the Hunter (film)',
                 'The Philadelphia Story (1940) (film)':'The Philadelphia Story (film)','The Replacements (2000) (film)':'The Replacements (film)','The Right Stuff (1983) (film)':'The Right Stuff (film)',
                 'The Sandlot (1993) (film)':'The Sandlot', 'The Treasure of the Sierra Madre (1948) (film)':"The Treasure of the Sierra Madre (film)",
                 'Three Kings (1999) (film)':'Three Kings (1999 film)','Topsy-Turvy (1999) (film)': 'Topsy-Turvy','True Grit (1969) (film)':'True Grit (1969 film)',
                 'True Grit (2010) (film)': 'True Grit (2010 film)','Trumbo (2007) (film)': 'Trumbo (2007 film)','Undefeated (2012) (film)': 'Undefeated (2011 film)',
                 'We Are What We Are (2013) (film)':'We Are What We Are (2013 film)', 'We Were Here (2011) (film)':'We Were Here (film)',
                 "What's Love Got To Do With It? (1993) (film)":"What's Love Got to Do with It (film)", 'Wild Wild West (1999) (film)':"Wild Wild West"}





# yeah, this part sucked.

# correct_rename

# rename_success_list = []
# rename_error_list = []
# for key, value in correct_rename.items():
#     try:
#         wiki_html_page = wikipedia.page(value)
#         rename_success_list.append((key, value, wiki_html_page))
#     except:
#         rename_error_list.append((key, value))
        

# for i in rename_success_list:
#     movie_dictionary[i[1]] = movie_dictionary[i[0]]
#     movie_dictionary[i[1]].append(i[2])
#     del movie_dictionary[i[0]]

# rename_error_list

# movie_dictionary[rename_error_list[0][0]].append("")
# movie_dictionary[rename_error_list[0][0]].append("1931-01-30")
# movie_dictionary[rename_error_list[0][0]].append(87.0)
# movie_dictionary[rename_error_list[0][0]].append("$5 million")
# movie_dictionary[rename_error_list[0][0]].append("$1.5 million")
# movie_dictionary[rename_error_list[0][0]].append("United States")
# movie_dictionary[rename_error_list[0][0]].append("English")

# movie_dictionary[rename_error_list[1][0]].append("")
# movie_dictionary[rename_error_list[1][0]].append("1999-12-15")
# movie_dictionary[rename_error_list[1][0]].append(160.0)
# movie_dictionary[rename_error_list[1][0]].append("$5.2 million")
# movie_dictionary[rename_error_list[1][0]].append(np.nan)
# movie_dictionary[rename_error_list[1][0]].append("United States")
# movie_dictionary[rename_error_list[1][0]].append("English")



