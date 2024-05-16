import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm_notebook
from sqlalchemy import create_engine, inspect
from datetime import datetime

bpcReq = requests.get("https://www.biopharmcatalyst.com/calendars/fda-calendar")

bpcSup = BeautifulSoup(bpcReq.text, "lxml")

eventDF = pd.DataFrame

for tr_el in tqdm_notebook(bpcSup.findAll('tr')[1:]): #start from row 1 to exclude header row
    row = tr_el.find_all('td')
    if 'pdufa' in row[3]['data-value']: #This currently only finds PDUFA dates, but it could be easily expanded
        evTck = row[0].find('a').text.strip() #leftmost column element, returns text
        evDay = datetime.strptime(row[4].find('time').text, '%m/%d/%Y') #
        print evDay.date(), evTck
        

