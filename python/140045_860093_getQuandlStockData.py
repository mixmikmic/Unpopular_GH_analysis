import numpy as np
import pandas as pd
import dill
import quandl
from sqlalchemy import create_engine, inspect
from tqdm import tqdm_notebook

tickSymbs = sorted(dill.load(open('setOfTickerSymbols.pkl', 'r')))

quandl_key_handle = open("quandl.apikey", "r")
quandl.ApiConfig.api_key = quandl_key_handle.read()
quandl_key_handle.close()

engine = create_engine('sqlite:///capstone.db')

dlFails = []
dlWins = []
for ticker in tqdm_notebook(tickSymbs):
    try:
        #tickerDF = quandl.get("WIKI/%s" % ticker)
        quandl.get("WIKI/%s" % ticker).to_sql(ticker, engine, if_exists = 'replace')
    except:
        dlFails.append(ticker)
    else:
        dlWins.append(ticker)

inspector = inspect(engine)
print len(inspector.get_table_names())
print inspector.get_table_names()

