import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
import re
get_ipython().magic('matplotlib inline')
get_ipython().magic('autosave 120')
get_ipython().magic('run helper_functions.py')

# regex_runtime = r" minutes"
# subset = ""

# who

# movie_db = unpickle_object("movie_database_final.pkl")

# movie_df = pd.DataFrame.from_items(movie_db.items(), 
#                             orient='index', 
#                             columns=['A','B','C','D',"E",'F',"G", 'H', "I", "J", "K", "L"])


# del movie_df['F']

# movie_df.columns = ['Rank_in_genre', "rotton_rating_(/10)", "No._of_reviews_rotton","Tomato_Freshness_(%)", "audience_rating_(/5)", "Date", "Runtime", "Box_office", "Budget", "Country","Language" ]

# movie_df.head()

# movie_df['Rank_in_genre'] = movie_df['Rank_in_genre'].apply(lambda x: x.strip("."))
# movie_df['Rank_in_genre'] = movie_df['Rank_in_genre'].apply(lambda x: float(x))
# movie_df['rotton_rating_(/10)'] = movie_df['rotton_rating_(/10)'].apply(lambda x: float(x))
# movie_df["No._of_reviews_rotton"] = movie_df['No._of_reviews_rotton'].apply(lambda x: int(x))
# movie_df['Tomato_Freshness_(%)'] = movie_df['Tomato_Freshness_(%)'].apply(lambda x: x.strip("%"))
# movie_df['Tomato_Freshness_(%)'] = movie_df['Tomato_Freshness_(%)'].apply(lambda x: float(x))
# movie_df['audience_rating_(/5)'] = movie_df['audience_rating_(/5)'].apply(lambda x: x.strip("/5"))
# movie_df['audience_rating_(/5)'] = movie_df['audience_rating_(/5)'].apply(lambda x: float(x))
# movie_df['Date'] = movie_df['Date'].apply(lambda x: dt.strptime(x, '%Y-%m-%d'))
# movie_df['Month'] = movie_df["Date"].apply(lambda x: x.month)
# movie_df['Runtime'] = movie_df['Runtime'].apply(lambda x: str(x))
# movie_df['Runtime'] = movie_df['Runtime'].apply(lambda x: x.strip())
# movie_df['Runtime'] = movie_df['Runtime'].apply(lambda x: re.sub(regex_runtime, subset, x))
# movie_df['Runtime'] = movie_df['Runtime'].apply(lambda x: int(x))

# movie_df['Box_office'].unique()

# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: str(x))

# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip())

# billions

# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" million"))

# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("\xa0"))

# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("$"))

# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" million<"))

# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("¥"))

# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("$"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" b"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("\xa0"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" million USD"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("  billion\n$159.4 million)"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("¥"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" billion\n$28"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" million<"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(".682.627.806"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" million\n£'"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip("million\n\n$"))
# movie_df['Box_office'] = movie_df['Box_office'].apply(lambda x: x.strip(" billion toman (Ira"))




# def replacer(array,expression, change): #good helper functin to quickly replace bad formatting
#     for index, value in enumerate(array):
#         if expression in value:
#             array[index] = array[index].replace(expression, change)
        

