get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
from bs4 import BeautifulSoup
import unicodedata
import locale
locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' )
import requests

import json
with open('candidate_pages.json') as data_file:    
    data = json.load(data_file)

dic2014 = {}
for k in data.keys():
    if 'http://myneta.info/ls2014/' in k:
        dic2014[k] = data[k]

dic2009 = {}
for k in data.keys():
    if 'http://myneta.info/ls2009/' in k:
        dic2009[k] = data[k]

dic2004 = {}
for k in data.keys():
    if 'http://myneta.info/loksabha2004/' in k:
        dic2004[k] = data[k]

ex = data['http://myneta.info/ls2014/candidate.php?candidate_id=7120']

def get_info_2014(text):
    dic = {}
    soup = BeautifulSoup(text, "html.parser")
    for  r in soup.findAll("a"):
        try:
            if r['name'] == "liabilities":
                table = r
                for r in table.findAll('td'):
                    try:

                        if r.get_text()=="Loans from Banks / FIs":
                            dic["Loans from Banks / FIs"] = r.findNext('b').get_text()

                        if r.get_text()=="Loans due to Individual / Entity":
                            dic["Loans due to Individual / Entity"] = r.findNext('b').get_text()

                        if r.get_text()=="Any other Liability":
                            dic["Any other Liability"] = r.findNext('b').get_text()

                        if r.get_text()=="Grand Total of Liabilities (as per affidavit)":
                            dic["Grand Total of Liabilities (as per affidavit)"] = r.findNext('b').get_text()

                        if r.get_text()=="Dues to departments dealing with government accommodation":
                            dic["Dues to departments dealing with government accommodation"] = r.findNext('b').get_text()

                        if r.get_text()=="Dues to departments dealing with supply of water":
                            dic["Dues to departments dealing with supply of water"] = r.findNext('b').get_text()

                        if r.get_text()=="Dues to departments dealing with supply of electricity":
                            dic["Dues to departments dealing with supply of electricity"] = r.findNext('b').get_text()

                        if r.get_text()=="Dues to departments dealing with telephones":
                            dic["Dues to departments dealing with telephones"] = r.findNext('b').get_text()

                        if r.get_text()=="Dues to departments dealing with supply of transport":
                            dic["Dues to departments dealing with supply of transport"] = r.findNext('b').get_text()

                        if r.get_text()=="Income Tax Dues":
                            dic["Income Tax Dues"] = r.findNext('b').get_text()

                        if r.get_text()=="Wealth Tax Dues":
                            dic["Wealth Tax Dues"] = r.findNext('b').get_text()

                        if r.get_text()=="Service Tax Dues":
                            dic["Service Tax Dues"] = r.findNext('b').get_text()

                        if r.get_text()=="Property Tax Dues":
                            dic["Property Tax Dues"] = r.findNext('b').get_text()

                        if r.get_text()=="Sales Tax Dues":
                            dic["Sales Tax Dues"] = r.findNext('b').get_text()

                        if r.get_text()=="Any Other Dues":
                            dic["Any Other Dues"] = r.findNext('b').get_text()

                        if r.get_text()=="Grand Total of all Govt Dues (as per affidavit)":
                            dic["Grand Total of all Govt Dues (as per affidavit)"] = r.findNext('b').get_text()

                        if r.get_text()=="Whether any other liabilities are in dispute, if so, mention the amount involved and the authority before which it is pending":
                            dic["Whether any other liabilities are in dispute, if so, mention the amount involved and the authority before which it is pending"] = r.findNext('b').get_text()

                        if r.get_text()=="Totals (Calculated as Sum of Values)":
                            dic["Totals (Calculated as Sum of Values)"] = r.findNext('b').get_text()
                    except:
                        pass
        except:
            pass
    return dic

get_info_2014(ex)

def clean_dic_2014(dic):
    for k in dic.keys():
        if dic[k]==' Nil ':
            dic[k]= np.nan
        else:
            try:
                tmp = unicodedata.normalize('NFKD',dic[k]).encode('ascii','ignore').split('Rs')[1].strip()
                dic[k] = locale.atof(tmp)
            except:
                pass
    return dic

clean_dic_2014(get_info(ex))

get_ipython().run_cell_magic('time', '', "tmplist = []\nfor k in dic2014.keys():\n    dic = get_info_2014(data[k])\n    dic['url'] = k\n    tmplist.append(clean_dic_2014(dic))\nlist2014 = tmplist")

def get_info_2009(text):
    dic = {}
    soup = BeautifulSoup(text, "html.parser")
    for  r in soup.findAll('td'):
        if r.get_text()=="Loans from Banks":
            dic["Loans from Banks"] = r.findNext('b').get_text()
                            
        if r.get_text()=="Loans from Financial Institutions":
            dic["Loans from Financial Institutions"] = r.findNext('b').get_text()

        if r.get_text()=="(a) Dues to departments dealing with government accommodation":
            dic["(a) Dues to departments dealing with government accommodation"] = r.findNext('b').get_text()

        if r.get_text()=="(b) Dues to departments dealing with supply of water":
            dic["(b) Dues to departments dealing with supply of water"] = r.findNext('b').get_text()

        if r.get_text()=="(c) Dues to departments dealing with supply of electricity":
            dic["(c) Dues to departments dealing with supply of electricity"] = r.findNext('b').get_text()

        if r.get_text()=="(d) Dues to departments dealing with telephones":
            dic["(d) Dues to departments dealing with telephones"] = r.findNext('b').get_text()

        if r.get_text()=="(e) Dues to departments dealing with supply of transport":
            dic["(e) Dues to departments dealing with supply of transport"] = r.findNext('b').get_text()

        if r.get_text()=="(f) Other Dues if any":
            dic["(f) Other Dues if any"] = r.findNext('b').get_text()

        if r.get_text()=="(i) (a) Income Tax including surcharge [Also indicate the assessment year upto which Income Tax Return filed.]":
            dic["(i) (a) Income Tax including surcharge [Also indicate the assessment year upto which Income Tax Return filed.]"] = r.findNext('b').get_text()

        if '(ii) Wealth Tax [Also indicate the assessment year upto which Wealth Tax return filed.]' in r.get_text():
            dic["(ii) Wealth Tax [Also indicate the assessment year upto which Wealth Tax return filed.]"] = r.findNext('b').get_text()
        
        if r.get_text()=="(iii) Sales Tax [Only in case proprietary business]":
            dic["(iii) Sales Tax [Only in case proprietary business]"] = r.findNext('b').get_text()
        
        if r.get_text()=="(iv) Property Tax":
            dic["(iv) Property Tax"] = r.findNext('b').get_text()
        
        if r.get_text()=="Totals":
            dic["Totals"] = r.findNext('b').get_text()
    return dic

def clean_dic_2009(dic):
    for k in dic.keys():
        if dic[k]=='Nil':
            dic[k]= np.nan
        else:
            try:
                tmp = unicodedata.normalize('NFKD',dic[k]).encode('ascii','ignore').split('Rs')[1].strip()
                dic[k] = locale.atof(tmp)
            except:
                pass
    return dic

ex = dic2009['http://myneta.info/ls2009/candidate.php?candidate_id=306']
clean_dic_2009(get_info_2009(ex))

get_ipython().run_cell_magic('time', '', "tmplist = []\nfor k in dic2009.keys():\n    dic = get_info_2009(data[k])\n    dic['url'] = k\n    tmplist.append(clean_dic_2009(dic))\nlist2009 = tmplist")

def get_info_2004(text):
    dic = {}
    soup = BeautifulSoup(text, "html.parser")
    for  r in soup.findAll('td'):
        if r.get_text()=="Loans from Banks":
            dic["Loans from Banks"] = r.findNext('b').get_text()
                            
        if r.get_text()=="Loans from Financial Institutions":
            dic["Loans from Financial Institutions"] = r.findNext('b').get_text()

        if r.get_text()=="(a) Dues to departments dealing with government accommodation":
            dic["(a) Dues to departments dealing with government accommodation"] = r.findNext('b').get_text()

        if r.get_text()=="(b) Dues to departments dealing with supply of water":
            dic["(b) Dues to departments dealing with supply of water"] = r.findNext('b').get_text()

        if r.get_text()=="(c) Dues to departments dealing with supply of electricity":
            dic["(c) Dues to departments dealing with supply of electricity"] = r.findNext('b').get_text()

        if r.get_text()=="(d) Dues to departments dealing with telephones":
            dic["(d) Dues to departments dealing with telephones"] = r.findNext('b').get_text()

        if r.get_text()=="(e) Dues to departments dealing with supply of transport":
            dic["(e) Dues to departments dealing with supply of transport"] = r.findNext('b').get_text()

        if r.get_text()=="(f) Other Dues if any":
            dic["(f) Other Dues if any"] = r.findNext('b').get_text()

        if r.get_text()=="(i) (a) Income Tax including surcharge [Also indicate the assessment year upto which Income Tax Return filed.]":
            dic["(i) (a) Income Tax including surcharge [Also indicate the assessment year upto which Income Tax Return filed.]"] = r.findNext('b').get_text()

        if '(ii) Wealth Tax [Also indicate the assessment year upto which Wealth Tax return filed.]' in r.get_text():
            dic["(ii) Wealth Tax [Also indicate the assessment year upto which Wealth Tax return filed.]"] = r.findNext('b').get_text()
        
        if r.get_text()=="(iii) Sales Tax [Only in case proprietary business]":
            dic["(iii) Sales Tax [Only in case proprietary business]"] = r.findNext('b').get_text()
        
        if r.get_text()=="(iv) Property Tax":
            dic["(iv) Property Tax"] = r.findNext('b').get_text()
        
        if r.get_text()=="Totals":
            dic["Totals"] = r.findNext('b').get_text()
    return dic

def clean_dic_2004(dic):
    for k in dic.keys():
        if dic[k]=='Nil':
            dic[k]= np.nan
        else:
            try:
                tmp = unicodedata.normalize('NFKD',dic[k]).encode('ascii','ignore').split('Rs')[1].strip()
                dic[k] = locale.atof(tmp)
            except:
                pass
    return dic

ex = dic2004['http://myneta.info/loksabha2004/candidate.php?candidate_id=3453']
clean_dic_2004(get_info_2004(ex))

get_ipython().run_cell_magic('time', '', "tmplist = []\nfor k in dic2004.keys():\n    dic = get_info_2004(data[k])\n    dic['url'] = k\n    tmplist.append(clean_dic_2004(dic))\nlist2004 = tmplist")

print len(data)
print len(list2004) + len(list2009) + len(list2014)

def change_dic2014(orig):
    dic = {}
    dic['url'] = orig['url']
    dic['liab_banks/fis'] = orig['Loans from Banks / FIs']
    dic['liab_accom'] = orig['Dues to departments dealing with government accommodation']
    dic['liab_water'] = orig['Dues to departments dealing with supply of water']
    dic['liab_elec'] = orig['Dues to departments dealing with supply of electricity']
    dic['liab_tel'] = orig['Dues to departments dealing with telephones']
    dic['liab_transp'] = orig['Dues to departments dealing with supply of transport']
    dic['liab_other'] = orig['Any other Liability']
    dic['liab_tax_income'] = orig['Income Tax Dues']
    dic['liab_tax_wealth'] = orig['Wealth Tax Dues']
    dic['liab_tax_sales'] = orig['Sales Tax Dues']
    dic['liab_tax_prop'] = orig['Property Tax Dues']
    dic['liab_total'] = orig['Totals (Calculated as Sum of Values)']
    return dic    

def change_dic2009(orig):
    dic = {}
    dic['url'] = orig['url']
    if np.isnan(orig['Loans from Banks']) and np.isnan(orig['Loans from Financial Institutions']):
        dic['liab_banks/fis'] = np.nan
    if np.isnan(orig['Loans from Banks'])==False and np.isnan(orig['Loans from Financial Institutions']):
        dic['liab_banks/fis'] = orig['Loans from Banks']
    if np.isnan(orig['Loans from Banks']) and np.isnan(orig['Loans from Financial Institutions'])==False:
        dic['liab_banks/fis'] = orig['Loans from Financial Institutions']
    dic['liab_accom'] = orig['(a) Dues to departments dealing with government accommodation']
    dic['liab_water'] = orig['(b) Dues to departments dealing with supply of water']
    dic['liab_elec'] = orig['(c) Dues to departments dealing with supply of electricity']
    dic['liab_tel'] = orig['(d) Dues to departments dealing with telephones']
    dic['liab_transp'] = orig['(e) Dues to departments dealing with supply of transport']
    dic['liab_other'] = orig['(f) Other Dues if any']
    dic['liab_tax_income'] = orig['(i) (a) Income Tax including surcharge [Also indicate the assessment year upto which Income Tax Return filed.]']
    dic['liab_tax_wealth'] = orig['(ii) Wealth Tax [Also indicate the assessment year upto which Wealth Tax return filed.]']
    dic['liab_tax_sales'] = orig['(iii) Sales Tax [Only in case proprietary business]']
    dic['liab_tax_prop'] = orig['(iv) Property Tax']
    dic['liab_total'] = orig['Totals']
    return dic 

def change_dic2004(orig):
    dic = {}
    dic['url'] = orig['url']
    if np.isnan(orig['Loans from Banks']) and np.isnan(orig['Loans from Financial Institutions']):
        dic['liab_banks/fis'] = np.nan
    if np.isnan(orig['Loans from Banks'])==False and np.isnan(orig['Loans from Financial Institutions']):
        dic['liab_banks/fis'] = orig['Loans from Banks']
    if np.isnan(orig['Loans from Banks']) and np.isnan(orig['Loans from Financial Institutions'])==False:
        dic['liab_banks/fis'] = orig['Loans from Financial Institutions']
    dic['liab_accom'] = orig['(a) Dues to departments dealing with government accommodation']
    dic['liab_water'] = orig['(b) Dues to departments dealing with supply of water']
    dic['liab_elec'] = orig['(c) Dues to departments dealing with supply of electricity']
    dic['liab_tel'] = orig['(d) Dues to departments dealing with telephones']
    dic['liab_transp'] = orig['(e) Dues to departments dealing with supply of transport']
    dic['liab_other'] = orig['(f) Other Dues if any']
    dic['liab_tax_income'] = orig['(i) (a) Income Tax including surcharge [Also indicate the assessment year upto which Income Tax Return filed.]']
    dic['liab_tax_wealth'] = orig['(ii) Wealth Tax [Also indicate the assessment year upto which Wealth Tax return filed.]']
    dic['liab_tax_sales'] = orig['(iii) Sales Tax [Only in case proprietary business]']
    dic['liab_tax_prop'] = orig['(iv) Property Tax']
    dic['liab_total'] = orig['Totals']
    return dic 

change_dic2009(list2009[1000])

change_dic2014(list2014[1000])

change_dic2004(list2004[1000])

allyears = []
for dic in list2004:
    allyears.append(change_dic2004(dic))
for dic in list2009:
    allyears.append(change_dic2009(dic))
for dic in list2014:
    allyears.append(change_dic2014(dic))

len(allyears)

liabilities = pd.DataFrame(allyears)
liabilities.to_csv("liabilities.csv", header=True, index=False)



