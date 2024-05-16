get_ipython().magic('matplotlib inline')
from bs4 import BeautifulSoup
import urllib2
import requests
import pandas as pd
import re
import time
import numpy as np
import json
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import matplotlib.pyplot as plt
from pyquery import PyQuery as pq

# Our source page: Lok Sabha 2014#
all_cand = "http://myneta.info/ls2014/index.php?action=summary&subAction=candidates_analyzed&sort=candidate#summary"
source=requests.get(all_cand) # Modify here: winners works with smaller sample, all_cand with the whole set
tree= BeautifulSoup(source.text,"html.parser")
table_pol = tree.findAll('table')[2]
rows = table_pol.findAll("tr")[2:]

# We build one dictionary per candidate #
# We give each candidates to idenfiers: its order in the list and its url address

def id2(r):
    url_string = str(r.find("a").get("href"))
    id2 = int(re.search(r'\d+', url_string).group())
    return id2

def c_link(r):
    return r.find("a").get("href")
def name(r):
    return r.find("a").get_text()
def cols(r):
    return r.findAll("td")
def assets(r):
    col = cols(r)
    ass1 = col[6].get_text().split("~")[0].encode('ascii', 'ignore').replace("Rs","").replace(",","")
    if ass1 == "Nil":
        ass2 = 0
    else:
        ass2=int(ass1)
    return ass2

def liab(r):
    col = cols(r)
    liab1 = col[7].get_text().split("~")[0].encode('ascii', 'ignore').replace("Rs","").replace(",","")
    if liab1 == "Nil":
        liab2 = 0
    else:
        liab2 = int(liab1)
    return liab2

info_candidate = lambda r: [int(cols(r)[0].get_text()),id2(r), r.find("a").get("href"),r.find("a").get_text(),
                            cols(r)[2].get_text(),cols(r)[3].get_text(),cols(r)[5].get_text(),
                            int(cols(r)[4].get_text()),assets(r), liab(r)]

title = ['id','id2','url','name','district','party','education','nr_crime','assets','liabilities']
dict_candidates = [dict(zip(title,info_candidate(r))) for r in rows]
print len(dict_candidates)

# Now we create a really big dictionary which stores url and page for each candidate
# Work in progress...
# First transform dict_candidate into a dataframe

df_pol = pd.DataFrame(dict_candidates)
df_pol.to_csv("C:\Users\mkkes_000\Dropbox\Indiastuff\OutputTables\df_pol_LS2014.csv", index = True)
order_cols = ['id2','name','district','party','education','assets','liabilities','nr_crime','url']
df_pol = df_pol[order_cols].sort(['assets'],ascending=0)

urlcache={}

def get_page(url):
    # Check if URL has already been visited.
    url_error = []
    if (url not in urlcache) or (urlcache[url]==1) or (urlcache[url]==2):
        time.sleep(1)
        steps = len(urlcache)
        if 100*int(steps/100)==steps:
            print steps # This counter tells us how many links were downloaded at every 100 mark
        # try/except blocks are used whenever the code could generate an exception (e.g. division by zero).
        # In this case we don't know if the page really exists, or even if it does, if we'll be able to reach it.
        try:
            r = requests.get("http://myneta.info/ls2014/%s" % url)
            if r.status_code == 200:
                urlcache[url] = r.text
            else:
                urlcache[url] = 1
        except:
            urlcache[url] = 2
            url_error.append(url)
            print url
    return urlcache[url]

# retry downloading missing pages:
for r in url_error:
    urlcache[r] = requests.get("http://myneta.info/ls2014/%s" % r).text()

#df_pol["url"].apply(get_page) # This is a very long call (~4.5 hours on full dataset)
                              # I am saving it in order to run it only once

print np.sum([(urlcache[k]==1) or (urlcache[k]==2) for k in urlcache])# no one or 0's
print len(df_pol.url.unique())==len(urlcache)#we got all of the urls

with open("tempdata/polinfo.json","w") as fd:
    json.dump(pol_pages, fd)
del urlcache

