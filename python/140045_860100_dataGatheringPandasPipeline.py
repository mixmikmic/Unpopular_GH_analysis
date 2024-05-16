import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm_notebook
import dill
from tqdm import tqdm_notebook
from alpha_vantage.timeseries import TimeSeries

from urllib2 import urlopen
import ics
import re
tickerRe = re.compile(r"\A[A-Z]{3,4}\W")
today = datetime.today()

FdaUrl = "https://calendar.google.com/calendar/ical/5dso8589486irtj53sdkr4h6ek%40group.calendar.google.com/public/basic.ics"

FdaCal = ics.Calendar(urlopen(FdaUrl).read().decode('iso-8859-1'))

FdaCal

past_pdufa_syms = set()
for event in tqdm_notebook(FdaCal.events):
    matches = re.findall(tickerRe, event.name)
    if len(matches) >=1:
        eComp = str(matches[0]).strip().strip(".")
        past_pdufa_syms.add(eComp)

av_key_handle = open("alphavantage.apikey", "r")
ts = TimeSeries(key=av_key_handle.read().strip(), output_format='pandas')
av_key_handle.close()

dataframes = dict()

fails = set()
wins = set()
for ticker in tqdm_notebook(past_pdufa_syms):
    try:
        df, meta = ts.get_daily(symbol=ticker, outputsize='full')
        dataframes[meta["2. Symbol"]] = df
    except:
        fails.add(ticker)
    else:
        wins.add(meta["2. Symbol"])

print len(fails), len(wins)

companies = dataframes.keys()

price_and_fda = dict()
for company in tqdm_notebook(companies):
    company_events = []
    for event in FdaCal.events:
        matches = re.findall(tickerRe, event.name)
        if len(matches)>=1:
            if company in matches[0]:
                #print company, event.name, event.begin
                company_events.append((event.begin.datetime.strftime("%Y-%m-%d"), True))
    price = dataframes[company]
    raw_dates = pd.DataFrame(company_events, columns = ["date", "pdufa?"])
    dates = raw_dates.set_index("date")
    #print dates
    #print price
    final = price.join(dates,rsuffix='_y')
    final['pdufa?'].fillna(value=False, inplace = True)
    price_and_fda[company] = final
    
                

price_and_fda['MRK']['pdufa?']

dill.dump(price_and_fda, open("Prices_and_FDA_Dates.pkl", "w"))



