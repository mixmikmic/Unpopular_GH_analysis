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
dropbox = "C:\Users\mkkes_000\Dropbox\Indiastuff\OutputTables"

with open(dropbox + "\candidate_pages.json") as json_file:
    pol_pages = json.load(json_file) # Next iterations will start from here!

def find_year(link):
    year = re.findall('\d+', link)
    return int(year[0])

find_year("http://www.myneta.info/andhra2014/index.php?action=affidavitComparison&myneta_folder2=ap09&id1=2150&id2=2415")

cleaner = lambda e: int(re.findall('\d+', e.replace(',', '').split(" ~ ")[0])[0])
income_cols = ["Relation","PAN","Year","Income"]

def income_table(candidate_id):
    page_candidate = pol_pages[candidate_id]
    c_soup = BeautifulSoup(page_candidate,"html.parser")
    table_titles =[x.get_text().strip() for x in c_soup.findAll("h3")]
    tables = [x.find_next() for x in c_soup.findAll("h3")]
    dict_tab = dict(zip(table_titles,tables))
    income_tab = dict_tab['Details of PAN and status of Income Tax return']
    income_rows = income_tab.find_all("tr")
    dict_income = {}
    df_inc = pd.DataFrame([])
    if income_cols==[]:
        dict_income = {'HH':{"Year":np.nan,"PAN":"N","Relation":np.nan,"Income":np.nan}}
    else:
        for r in income_rows[1:]:
            list_items = [x.get_text() for x in r.findAll("td")]
            if len(list_items)==4 and list_items[3]!="Nil":
                list_items[3] = cleaner(list_items[3])
            if len(list_items)==4 and list_items[3]=="Nil":
                list_items[3] = 0
            dict_income[list_items[0]] = dict(zip(income_cols,list_items))
        df_inc = df_inc.from_dict(dict_income,orient = "index")
    try:
        df_inc = df_inc[df_inc.PAN=="Y"]
        HHinc = np.sum(df_inc['Income'])
        HHDeclarations = np.count_nonzero(df_inc['PAN'])
        self_income = dict_income['self']['Income']
        self_declare = dict_income['self']['PAN']
    except AttributeError:
        df_inc=df_inc
        HHinc = np.nan
        HHDeclarations = 0
        self_income = np.nan
        self_declare = np.nan
    newdict = {'self_inc':self_income,'self_declare':self_declare,'HHinc':HHinc,"HHDeclarations":HHDeclarations}
    return newdict

candidate_id = pol_pages.keys()[5]
print candidate_id
page_candidate = pol_pages[candidate_id]
c_soup = BeautifulSoup(page_candidate,"html.parser")
table_titles =[x.get_text().strip() for x in c_soup.findAll("h3")]
tables = [x.find_next() for x in c_soup.findAll("h3")]
dict_tab = dict(zip(table_titles,tables))
dict_tab.keys()
#    income_tab = dict_tab['Details of PAN and status of Income Tax return']
#    income_rows = income_tab.find_all("tr")
#    dict_income = {}
#    df_inc = pd.DataFrame([])
#    if income_cols==[]:
#        dict_income = {'HH':{"Year":np.nan,"PAN":"N","Relation":np.nan,"Income":np.nan}}
#    else:
#        for r in income_rows[1:]:
#            list_items = [x.get_text() for x in r.findAll("td")]
#            if len(list_items)==4 and list_items[3]!="Nil":
#                list_items[3] = cleaner(list_items[3])
#            if len(list_items)==4 and list_items[3]=="Nil":
#                list_items[3] = 0
#            dict_income[list_items[0]] = dict(zip(income_cols,list_items))
#        df_inc = df_inc.from_dict(dict_income,orient = "index")
#    try:
   #     df_inc = df_inc[df_inc.PAN=="Y"]
    #    HHinc = np.sum(df_inc['Income'])
     #   HHDeclarations = np.count_nonzero(df_inc['PAN'])
      #  self_income = dict_income['self']['Income']
       # self_declare = dict_income['self']['PAN']
   # except AttributeError:
    #    df_inc=df_inc
     #   HHinc = np.nan
      #  HHDeclarations = 0
       # self_income = np.nan
        #self_declare = np.nan
    #newdict = {'self_inc':self_income,'self_declare':self_declare,'HHinc':HHinc,"HHDeclarations":HHDeclarations}

get_ipython().run_cell_magic('time', '', 'counterror = 0\ndict_allinc = {}\nif any("Details of PAN and status of Income Tax return" in dict_tab.keys())==False:\n    break\nelse:\n    for k,cid in enumerate(pol_pages.keys()):\n        year = find_year(cid)\n        try:\n            dict_allinc[cid] = income_table(cid)\n        except TypeError:\n            counterror = counterror+1\n            print "Error with this page: ", cid\n        except KeyError:\n            counterror = counterror+1\n            print "Error with this page: ", cid\n        if k%100==0:\n            print k,\nprint "\\n Number of errors: ", counterror')

d_inc_HH=pd.DataFrame([])
d_inc_HH = d_inc_HH.from_dict(dict_allinc,orient = "index") #d_inc_HH associates income to all candidates
                                                             # as well as for the whole family
                                                             # and the number of declarations
d_inc_HH.to_csv("C:\Users\mkkes_000\Dropbox\Indiastuff\OutputTables\incomes.csv", index=True)
d_inc_HH.head()

#This part is just to play a bit with incomes
d_inc = d_inc_HH.copy()
d_inc['ln_HHinc'] = np.log(d_inc['HHinc'])
d_inc['ln_selfinc'] = np.log(d_inc['self_inc'])
d_inc['sh_self'] = d_inc['self_inc']/d_inc['HHinc']
sns.kdeplot(d_inc.ln_HHinc)
sns.kdeplot(d_inc.ln_selfinc)

d_inc[d_inc.HHinc!=0].describe()

cols = ['year','cid','full_name','district','state','party_full','address','self_profession','spouse_profession','age']
def personal_info(candidate_id):
    year = find_year(candidate_id)
    page_candidate = pol_pages[candidate_id]
    c_soup = BeautifulSoup(page_candidate,"html.parser")
    personal = c_soup.findAll(attrs={"class": "grid_3 alpha"})[0]
    full_name = personal.find("h2").get_text().strip().title()
    district1 = personal.find("h5").get_text().strip()
    district = district1.title()
    state = district1[district1.find("(")+1:district1.find(")")].title()
    grid2 = personal.findAll(attrs={"class":"grid_2 alpha"})
    party_full = grid2[0].get_text().split(":")[1].split("\n")[0]
    age = grid2[2].get_text().split(":")[1].split("\n")[0]
    try:
        age = float(age)
    except ValueError:
        age = np.nan
    address = grid2[3].get_text().split(":")[1].split("\n")[1].strip() # Careful this one changes
    if personal.find("p").get_text()=="":
        self_profession = ""
        spouse_profession = ""
    else:
        self_profession = personal.find("p").get_text().split('\n')[0].split(":")[1].capitalize()
        spouse_profession = personal.find("p").get_text().split('\n')[1].split(":")[1].capitalize()
    list_info = [candidate_id,full_name,district,state,party_full,address,self_profession,spouse_profession]
    list_encode = [year]+[x.encode('utf-8') for x in list_info]+[age]
    dict_info = dict(zip(cols,list_encode))
    return dict_info

get_ipython().run_cell_magic('time', '', 'counterror = 0\ndict_allcand = {}\nfor k,cid in enumerate(pol_pages.keys()):\n    try:\n        dict_allcand[cid] = personal_info(cid)\n    except TypeError:\n        counterror = counterror+1\n        print "Error with this page: ", cid\n    if k%100==0:\n        print k,\nprint "Number of errors: ", counterror')

d_perso_info = pd.DataFrame([])
d_perso_info = d_perso_info.from_dict(dict_allcand, orient="index") # Dumping into a dataframe
d_perso_info.to_csv("C:\Users\mkkes_000\Dropbox\Indiastuff\OutputTables\info_perso_LS.csv",index = True)

d_perso_info[d_perso_info.self_profession!=""].head(10)

d_perso_info.describe()

sns.kdeplot(d_perso_info.age.dropna(),cumulative=True)

