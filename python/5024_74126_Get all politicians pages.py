get_ipython().magic('matplotlib inline')
import itertools
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
dropbox = "C:\Users\mkkes_000\Dropbox\Indiastuff\OutputTables"

#First load the dataframe
df_fin = pd.read_csv("df_all_regional_elections2.csv")
print df_fin.shape
df_fin.head()
df_fin.columns

def url_tsform(el,c_id):
    url_link = "http://myneta.info/%s/candidate.php?candidate_id=%s" %(el,c_id)
    return url_link
def url_split(url_link):
    return url_link.split("/")[3], url_link.split("=")[1]

url_split("http://myneta.info/mp2013/candidate.php?candidate_id=666")

def get_page(el,c_id):
    # Check if URL has already been visited.
    url = (el,c_id)
    url_error = []
    if (url not in urlcache) or (urlcache[url]==1) or (urlcache[url]==2):
        time.sleep(.5)
        try:
            r = requests.get(url_tsform(el,c_id))
            if r.status_code == 200:
                urlcache[url] = r.text
            else:
                urlcache[url] = 1
        except:
            urlcache[url] = 2
            url_error.append(url)
            print "error with:", url
    return urlcache[url]

#Re open the dict
with open('tempdata/largedict.json', 'r') as f:
     dict_start = json.load(f)

dict_id = {url_split(k):v for k,v in dict_start.iteritems()}
print len(dict_id)
urlcache = {}

#Creates a giant dictionary
#dict_id2 = {}
steps = len(dict_id2)
print "initial length: ", steps
for row in df_fin.itertuples():
    if (row[19],row[20]) in dict_id2:
        pass
    else:
        dict_id2[(row[19],row[20])] = get_page(row[19],row[20])
        steps = len(dict_id2)
        if steps % 1000 ==0:
            print steps,

dict_clean = {url_tsform(k[0],k[1]):v for k,v in dict_id2.iteritems() if v!=2}
print len(dict_clean), len(dict_id2)

with open('tempdata/largedict.json', 'w') as f:
     json.dump(dict_clean, f)

listerror=[]
for k,v in dict_id2.iteritems():
    if isinstance(v, basestring):
        listerror.append(v)
len(listerror)

page_ex = dict_clean['http://myneta.info/mp2013/candidate.php?candidate_id=258']

bs = BeautifulSoup(page_ex,"html.parser")
bs.findAll("table")[0]

ROOT_LINK = "http://myneta.info"
def get_otherelec_link(page_text, url):
    for a in pq(page_text)("a"):
        if a.text == "Click here for more details":
            other_elec_link = pq(a).attr.href
            return ROOT_LINK+other_elec_link
    return False

test_index = "http://myneta.info/mp2013/candidate.php?candidate_id=258"
page_data = dict_clean[test_index]
other_link = get_otherelec_link(page_data,test_index)

def get_otherelec_data(otherelec_link):
    
    otherelec_dict = {'common_link': otherelec_link}
    
    html = requests.get(otherelec_link)
    doc = pq(html.content)
    
    columns = []
    all_dicts = []
    add = 0
    trs = doc('tr')
    for tr in trs:
        elec_dict = otherelec_dict.copy()
        for th in pq(tr)('th'):
            columns.append(pq(th).text().replace(" ","_"))
            add = 0
        for i,td in enumerate(pq(tr)('td')):
            a = pq(td)('a')
            if a:
                elec_dict['elec_link'] = ROOT_LINK+a.attr.href
                elec_dict[columns[i]] = a.text()
            else:
                try:
                    if pq(td)('span') and i < 6:
                        elec_dict[columns[i]] = pq(td)('span').text()
                    else:
                        elec_dict[columns[i]] = str(pq(td).contents()[0]).encode('utf-8').strip().replace(',','')
                except:
                    print ""
                    print "Skipping col %s for %s" % (columns[i], elec_dict['common_link'])
            add = 1
            
        if add == 1:
            all_dicts.append(elec_dict)
    
    return all_dicts

get_otherelec_data(other_link)

def get_all_elec():
    all_elec_data = []   
    counter = 0.0
    for key, val in dict_clean.iteritems():
        thelink = get_otherelec_link(val,key)
        counter += 1
        if counter%100 == 0.0:
            print ".",
        if thelink:
            all_elec_data = all_elec_data + get_otherelec_data(thelink)
    
    df = pd.DataFrame(all_elec_data)
    return df.drop_duplicates()

all_elecs_df = get_all_elec()
all_elecs_df.head(3)['common_link']

all_elecs_df.shape

all_elecs_df.to_excel("tempdata/all_elecs_statelevel.xls")



