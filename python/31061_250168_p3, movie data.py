import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

movie = pd.read_csv("movies.csv")

movie.head(2)

movie.info()

sum(movie["release_year"]>=2000) 

movie = movie [movie["release_year"]>=2000]

sns.boxplot(movie["vote_average"])

plt.figure (figsize=(10,5))
ax = sns.countplot(movie["release_year"])
plt.xticks(rotation=90)
# ax.set_xticklabels(movie["release_year"], rotation=90,fontsize=15)
plt.show()

num = movie.shape[0]

genres = movie['genres'].unique() # len: 2040, due to remix
movie['genres'].value_counts()[0:10]

data = movie.loc[:,['id','release_year','genres']]

data['genres']=data['genres'].apply(lambda s: str(s).split("|")[0])

data.head()

sns.countplot(x = "release_year", hue = "genres", data = data)

gb = data.groupby(["release_year", "genres"]).count()

gb.head()

attributes = ["budget_adj", "revenue_adj", "tagline", "keywords","genres","production_companies", "vote_average","release_year"]
data = movie[attributes]

data.info()

data["production_companies"]=data["production_companies"].apply(lambda s: str(s).split("|")[0])

data["production_companies"].value_counts()[0:10]

two = data[(data["production_companies"] == "Paramount Pictures")| 
           (data["production_companies"] =="Universal Pictures") ]

two1 = two[(two["budget_adj"]>0) & (two["revenue_adj"]>0)]  # remove missing data

two1.info()

two1.groupby("production_companies").median()

two1["production_companies"].value_counts()

data = movie[movie["keywords"].notnull()]  # 5904
cnt = 0
total = data.shape[0]
for i,keyword in data['keywords'].iteritems():
    if "based" in keyword:
        cnt += 1
        #print(cnt,i,keyword)
print(cnt,total)
#  since 1960 /2000
# "novel" 295 /193
# "true" 58
# "based" 495 /344
# "nudity" 283/151

attributes = ["budget_adj", "revenue_adj", "tagline", "keywords","genres","production_companies", "vote_average","release_year"]
data = data[attributes]

data ['novel'] = data['keywords'].apply(lambda x: "novel" in x)

data1 = data[(data["budget_adj"]>0) & (data["revenue_adj"]>0)] 

data1.groupby('novel').median()

data1['novel'].value_counts()

130/2260

movie.info()

movie[movie["original_title"]== "Avatar"]

data['genres']=data['genres'].apply(lambda s: str(s).split("|")[0])

data [data['genres']== "Animation"].info()



