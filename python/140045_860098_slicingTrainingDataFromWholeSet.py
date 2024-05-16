import dill
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

company_dataframes = dill.load(open('Prices_and_FDA_Dates.pkl', 'r'))
company_list = company_dataframes.keys()

len(company_list)

closing_index = dill.load(open("close_price_stats_frame.pkl", "r"))

closing_index

company_dataframes['NEW'].loc[company_dataframes['NEW']['pdufa?'] == True]

testdf = company_dataframes['NEW']

testdf.reset_index(inplace = True)

testdf.loc[testdf['pdufa?']]

testdf.loc[1699]

testdf.loc[1699]['close']

testind = testdf.index[testdf['pdufa?'] == True]

testind

testind[0]

company_dataframes['ABBV'].join(closing_index, how='left')

data = []
for company in tqdm_notebook(company_list):
    df = company_dataframes[company].reset_index()
    pdufa_dates = df.index[df['pdufa?']].tolist()
    if len(pdufa_dates) > 0:
        for date in pdufa_dates:
            pRange = range(date-120, date)
            fRange = range(date, date+121)
            pCloses, pVolumes, fCloses, fVolumes = [], [], [], []
            for i in pRange:
                try:
                    pCloses.append(df.loc[i]['close'])
                    pVolumes.append(df.loc[i]['volume'])
                except:
                    pCloses.append(None)
                    pVolumes.append(None)
            for i in fRange:
                try:
                    fCloses.append(df.loc[i]['close'])
                    fVolumes.append(df.loc[i]['volume'])
                except:
                    fCloses.append(None)
                    fVolumes.append(None)
            data.append((company, df.loc[date]['index'], (pCloses, pVolumes), (fCloses, fVolumes)))

dill.dump(data, open('stock_price_training_slices.pkl', 'w'))

norm_data = []
for company in tqdm_notebook(company_list):
    df = company_dataframes[company].join(closing_index, how='left').reset_index()
    pdufa_dates = df.index[df['pdufa?']].tolist()
    if len(pdufa_dates) > 0:
        for date in pdufa_dates:
            pRange = range(date-120, date-7)
            pCloses, pVolumes = [], []
            for i in pRange:
                try:
                    close_price = df.loc[i]['close']
                    volume = df.loc[i]['volume']
                    mean_price = df.loc[i]['CP_mean']
                    stdv_price = df.loc[i]['CP_stdv']
                    pCloses.append(( df.loc[i]['index'],(close_price-mean_price)/(stdv_price) ))
                    pVolumes.append(( df.loc[i]['index'], volume ))
                except:
                    pCloses.append(None)
                    pVolumes.append(None)
            norm_data.append((company, df.loc[date]['index'], (pCloses, pVolumes)))

norm_data[:2]

scores = [line.split() for line in open("score_sheet_complete.txt", "r").readlines()]

scores[:2]

norm_data_annotated = []
for datum, score in zip(norm_data, scores):
    if datum[0] == score [0] and datum [1] == score[1]:
        norm_data_annotated.append((datum[0], datum[1], 
                                    score[2], datum[2] ))
    else:
        print "whoops theres a mismatch"
        

norm_data_annotated[:2]

dill.dump(norm_data_annotated, open('normalized_stock_price_slices.pkl', 'w'))

