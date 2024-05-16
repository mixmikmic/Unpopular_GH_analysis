import dill
import numpy as np
import pandas as pd

company_dataframes = dill.load(open('Prices_and_FDA_Dates.pkl', 'r'))

df1 = company_dataframes['AAAP']
df1_small = df1.loc['2016-06-12':'2016-06-25']
df2 = company_dataframes['ABBV']
df2_small = df2.loc['2016-06-12':'2016-06-25']
df2.columns = ['a','b','c','d','e','f']

first = True
for ticker, comp_df in company_dataframes.iteritems():
    if first:
        market_df = comp_df
        market_df.columns = ["volume-"+ticker,
                             "close-"+ticker,
                             "high-"+ticker,
                             "open-"+ticker,
                             "low-"+ticker,
                             "pdufa-"+ticker]
        first = False
    else:
        #comp_df.columns = ["volume-"+ticker,"close-"+ticker,"high-"+ticker,"open-"+ticker,"low-"+ticker,"pdufa-"+ticker]
        market_df = pd.merge(market_df, comp_df, how='outer', left_index=True, right_index=True, suffixes=('', '-'+ticker))

price_mean = market_df.filter(regex='close').mean(axis = 1, skipna = True)
price_stdv = market_df.filter(regex='close').std(axis = 1, skipna = True)

stats_df = pd.merge(price_mean.to_frame(),
                    price_stdv.to_frame(), 
                    left_index=True, 
                    right_index=True, 
                    how='inner')
stats_df.rename(columns={u'0_x':"CP_mean", u'0_y':"CP_stdv"}, inplace=True)

stats_df

result = pd.merge(market_df, stats_df, 
                  left_index=True, right_index=True, how='inner')

result

dill.dump(result, open("dataframe_with_mean_stdv_price.pkl", "w"))

dill.dump(stats_df, open("close_price_stats_frame.pkl", "w"))

