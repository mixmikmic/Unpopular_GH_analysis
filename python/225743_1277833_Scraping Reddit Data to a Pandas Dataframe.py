import pandas as pd
import urllib.request as ur
import json
import time

#Header to be submitted to reddit.
hdr = {'User-Agent': 'codingdisciple:playingwithredditAPI:v1.0 (by /u/ivis_reine)'}

#Link to the subreddit of interest.
url = "https://www.reddit.com/r/datascience/.json?sort=top&t=all"

#Makes a request object and receive a response.
req = ur.Request(url, headers=hdr)
response = ur.urlopen(req).read()

#Loads the json data into python.
json_data = json.loads(response.decode('utf-8'))

#The actual data starts.
data = json_data['data']['children']

for i in range(10):
    #reddit only accepts one request every 2 seconds, adds a delay between each loop
    time.sleep(2)
    last = data[-1]['data']['name']
    url = 'https://www.reddit.com/r/datascience/.json?sort=top&t=all&limit=100&after=%s' % last
    req = ur.Request(url, headers=hdr)
    text_data = ur.urlopen(req).read()
    datatemp = json.loads(text_data.decode('utf-8'))
    data += datatemp['data']['children']
    print(str(len(data))+" posts loaded")
    

#Create a list of column name strings to be used to create our pandas dataframe
data_names = [value for value in data[0]['data']]
print(data_names)

#Create a list of dictionaries to be loaded into a pandas dataframe
df_loadinglist = []
for i in range(0, len(data)):
    dictionary = {}
    for names in data_names:
        try:
            dictionary[str(names)] = data[i]['data'][str(names)]
        except:
            dictionary[str(names)] = 'None'
    df_loadinglist.append(dictionary)
df=pd.DataFrame(df_loadinglist)

df.shape

df.tail()

#Counts each word and return a pandas series

def word_count(df, column):
    dic={}
    for idx, row in df.iterrows():
        split = row[column].lower().split(" ")
        for word in split:
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1
    dictionary = pd.Series(dic)
    dictionary = dictionary.sort_values(ascending=False)
    return dictionary

top_counts = word_count(df, "selftext")
top_counts[0:5]



