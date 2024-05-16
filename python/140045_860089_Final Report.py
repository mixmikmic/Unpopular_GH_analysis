from urllib2 import urlopen
import ics
import re
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
import dill
from random import randint

tickerRe = re.compile(r"\A[A-Z]{3,4}\W")
today = datetime.today()

FdaUrl = "https://calendar.google.com/calendar/ical/5dso8589486irtj53sdkr4h6ek%40group.calendar.google.com/public/basic.ics"
FdaCal = ics.Calendar(urlopen(FdaUrl).read().decode('iso-8859-1'))
FdaCal

past_pdufa_syms = set()
for event in FdaCal.events:
    matches = re.findall(tickerRe, event.name)
    if len(matches) >=1:
        eComp = str(matches[0]).strip().strip(".")
        past_pdufa_syms.add(eComp)

print past_pdufa_syms

av_key_handle = open("alphavantage.apikey", "r")
ts = TimeSeries(key=av_key_handle.read().strip(), output_format='pandas')
av_key_handle.close()

dataframes = dict()
value_errors = set()
other_errors = set()
for ticker in tqdm_notebook(past_pdufa_syms):
    try:
        df, meta = ts.get_daily(symbol=ticker, outputsize='full')
        dataframes[meta["2. Symbol"]] = df
    except ValueError:
        value_errors.add(ticker)
    except:
        other_errors.add(ticker)

print value_errors
print other_errors

dill.dump(dataframes, open('final_raw_dataframe_dict.pkl', 'w'))

dataframes = dill.load(open('final_raw_dataframe_dict.pkl', 'r'))

company_list = dataframes.keys()

price_and_fda = dict()
for company in tqdm_notebook(company_list):
    company_events = []
    for event in FdaCal.events:
        matches = re.findall(tickerRe, event.name)
        if len(matches)>=1:
            if company in matches[0]:
                company_events.append((event.begin.datetime.strftime("%Y-%m-%d"), True))
    price = dataframes[company]
    raw_dates = pd.DataFrame(company_events, columns = ["date", "pdufa?"])
    dates = raw_dates.set_index("date")
    final = price.join(dates,rsuffix='_y')
    final['pdufa?'].fillna(value=False, inplace = True)
    price_and_fda[company] = final

price_and_fda['ENTA'].head(3)

dill.dump(price_and_fda, open("final_Prices_and_PDUFAs.pkl", "w"))

price_and_fda = dill.load(open("final_Prices_and_PDUFAs.pkl", "r"))

first = True
for ticker, comp_df in price_and_fda.iteritems():
    if first:
        market_df = comp_df.copy()
        market_df.columns = ["volume-"+ticker,
                             "close-"+ticker,
                             "high-"+ticker,
                             "open-"+ticker,
                             "low-"+ticker,
                             "pdufa?-"+ticker]
        first = False
    else:
        market_df = pd.merge(market_df, comp_df, how='outer', left_index=True, right_index=True, suffixes=('', '-'+ticker))

price_mean = market_df.filter(regex='close').mean(axis = 1, skipna = True)
price_stdv = market_df.filter(regex='close').std(axis = 1, skipna = True)

stats_df = pd.merge(price_mean.to_frame(),
                    price_stdv.to_frame(), 
                    left_index=True, 
                    right_index=True, 
                    how='inner')
stats_df.rename(columns={u'0_x':"CP_mean", u'0_y':"CP_stdv"}, inplace=True)

stats_df.head()

dill.dump(stats_df, open("close_price_stats_frame_final.pkl", "w"))

stats_df = dill.load(open("close_price_stats_frame_final.pkl", "r"))

norm_data = []
for company in tqdm_notebook(company_list):
    df = price_and_fda[company].join(stats_df, how='left').reset_index()
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

scores = [line.split() for line in open("score_sheet_complete.txt", "r").readlines()]

norm_data_annotated = []
mismatches = []
for datum in tqdm_notebook(norm_data):
    for score in scores:
        if datum[0] == score [0] and datum [1] == score[1]:
            norm_data_annotated.append((datum[0], datum[1], score[2], datum[2] ))
            break

dill.dump(norm_data_annotated, open("normalized_training_data.pkl", "w"))

norm_data_annotated = dill.load(open("normalized_training_data.pkl", "r"))

def assemble_frame(datum):
    df = pd.DataFrame(datum[3][0], columns=['date','norm_price'])
    df['event'] = datum[0]+"/"+datum[1]
    df['outcome'] = int(datum[2])
    return df

first = True

for line in tqdm_notebook(norm_data_annotated):
    try:
        if first:
            agg_data = assemble_frame(line)
            first = False
        else:
            tmp_data = assemble_frame(line)
            agg_data = pd.concat([agg_data, tmp_data],ignore_index=True)
    except:
        print line[0], line[1], "failed"

agg_data['date_stamp'] = pd.to_datetime(agg_data['date'])
event_labels = pd.factorize(agg_data['event'])
agg_data["event_stamp"] = event_labels[0]

agg_data['null'] = pd.isnull(agg_data).apply(lambda x: sum(x) , axis=1)
cleaned_agg = agg_data[agg_data['null'] == 0]

cleaned_agg.head()

dill.dump(cleaned_agg, open('final_cleaned_price_slices.pkl', 'w'))

cleaned_agg = dill.load(open('final_cleaned_price_slices.pkl', 'r'))

from sklearn.cross_validation import train_test_split

train_data, test_data = train_test_split(norm_data_annotated, train_size = .9)

first = True

for line in tqdm_notebook(train_data):
    try:
        if first:
            train_df = assemble_frame(line)
            first = False
        else:
            tmp_df = assemble_frame(line)
            train_df = pd.concat([train_df, tmp_df],ignore_index=True)
    except:
        print line[0], line[1], "failed"

train_df['date_stamp'] = pd.to_datetime(train_df['date'])
event_labels = pd.factorize(train_df['event'])
train_df["event_stamp"] = event_labels[0]

train_df['null'] = pd.isnull(train_df).apply(lambda x: sum(x) , axis=1)
train_clean = train_df[train_df['null'] == 0]

first = True

for line in tqdm_notebook(test_data):
    try:
        if first:
            test_df = assemble_frame(line)
            first = False
        else:
            tmp_df = assemble_frame(line)
            test_df = pd.concat([test_df, tmp_df],ignore_index=True)
    except:
        print line[0], line[1], "failed"
test_df['date_stamp'] = pd.to_datetime(test_df['date'])
event_labels = pd.factorize(test_df['event'])
test_df["event_stamp"] = event_labels[0]

test_df['null'] = pd.isnull(test_df).apply(lambda x: sum(x) , axis=1)
test_clean = test_df[test_df['null'] == 0]

dill.dump(train_clean, open("final_train_df.pkl", "w"))
dill.dump(test_clean, open("final_test_df.pkl", "w"))

train_clean = dill.load(open("final_train_df.pkl", "r"))
test_clean = dill.load(open("final_test_df.pkl", "r"))

from tsfresh import extract_features

train_feats = extract_features(train_clean[['norm_price', 'event_stamp', 'date_stamp']], 
                              column_id="event_stamp", column_sort="date_stamp", 
                              column_value="norm_price", n_jobs=0).dropna(axis=1)

train_feats.head()

train_y =train_df[['event_stamp', 'outcome']].groupby('event_stamp').head(1).set_index('event_stamp')['outcome']

train_y.head()

test_feats = extract_features(test_clean[['norm_price', 'event_stamp', 'date_stamp']], 
                              column_id="event_stamp", column_sort="date_stamp", 
                              column_value="norm_price", n_jobs=0).dropna(axis=1)

test_feats.shape

test_y =test_df[['event_stamp', 'outcome']].groupby('event_stamp').head(1).set_index('event_stamp')['outcome']

test_y.shape

dill.dump(train_feats, open('final_train_features.pkl','w'))
dill.dump(test_feats, open('final_test_features.pkl','w'))

train_feats = dill.load(open("final_train_features.pkl", "r"))
test_feats = dill.load(open("final_test_features.pkl", "r"))

print"\n".join(list(train_feats.columns.values))

features_of_interest = ['norm_price__mean',
                        'norm_price__median',
                        'norm_price__mean_change',
                        #'norm_price__mean_abs_change',
                        'norm_price__first_location_of_maximum',
                        'norm_price__first_location_of_minimum',
                        'norm_price__linear_trend__attr_"slope"',
                        'norm_price__count_above_mean',
                        'norm_price__count_below_mean'
                       ]

print train_feats[features_of_interest].shape
train_feats[features_of_interest].head()

print test_feats[features_of_interest].shape
test_feats[features_of_interest].head()

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score

scaler = StandardScaler()
classifier = SVC(C=1, coef0=1, degree=1)
params = {"C":range(1,5),
          "degree":range(1,3),
          "coef0":range(1,3)
         }
classifier_gs = GridSearchCV(classifier, params)

classifier_gs.fit(scaler.fit_transform(train_feats), train_y)

classifier_gs.best_params_

cross_val_score(classifier, scaler.transform(test_feats), y=test_y)

dill.dump(classifier, open("final_trained_svc.pkl","w"))

classifier = dill.load(open("final_trained_svc.pkl","r"))

all_feats = extract_features(cleaned_agg[['norm_price', 'event_stamp', 'date_stamp']], 
                              column_id="event_stamp", column_sort="date_stamp", 
                              column_value="norm_price", n_jobs=0).dropna(axis=1)

cleaned_agg

print all_feats[features_of_interest].shape
all_feats[features_of_interest].head()

all_y =cleaned_agg[['event_stamp', 'outcome']].groupby('event_stamp').head(1).set_index('event_stamp')['outcome']

all_events =cleaned_agg[['event_stamp','event']].groupby('event_stamp').head(1).set_index('event_stamp')['event']

all_predictions = classifier_gs.predict(scaler.transform(all_feats))

events_and_predictions = pd.DataFrame(all_events).join(pd.DataFrame(all_predictions))

events_and_predictions.shape

random_guesses = np.random.randint(0,2,size=(events_and_predictions.shape[0], 1000000))

print random_guesses.shape
random_guesses

random_guess_means = np.mean(random_guesses, axis = 1)

random_guess_means.shape

events_and_predictions.columns = ['event', 'pass_prediction']
events_and_predictions['pass_random'] = [int(x) for x in random_guess_means.round()]
events_and_predictions['pass_all'] = 1

events_and_predictions.head(3)

predicted_passes = events_and_predictions[events_and_predictions['pass_prediction'] == 1]
predicted_fails = events_and_predictions[events_and_predictions['pass_prediction'] == 0]

def x_days_later(date_str, x):
    pdufa_day = datetime.strptime(date_str,"%Y-%m-%d")
    change = timedelta(days = x)
    delta_date = pdufa_day + change
    return delta_date.strftime("%Y-%m-%d")

lead_price = 7 #how many days before the PDUFA to sample the price history
lag_price = 60 #how many days after the PDUFA to sample the price history
prior_and_post_prices = []
mindate = datetime(9999, 12, 31)
maxdate = datetime(1, 1, 1)
for stamp in events_and_predictions['event']:
    ticker, date = stamp.split("/")
    if datetime.strptime(date,"%Y-%m-%d") < mindate:
        mindate = datetime.strptime(date,"%Y-%m-%d")
    if datetime.strptime(date,"%Y-%m-%d") > maxdate:
        maxdate = datetime.strptime(date,"%Y-%m-%d")
    try:
        p_7_day = price_and_fda[ticker].loc[x_days_later(date, -1*lead_price)]['close']
    except KeyError:
        p_7_day = None
    try:
        p_60_day = price_and_fda[ticker].loc[x_days_later(date,lag_price)]['close']
    except KeyError:
        try:
            p_60_day = price_and_fda[ticker].loc[x_days_later(date,lag_price-1)]['close']
        except KeyError:
            try:
                p_60_day = price_and_fda[ticker].loc[x_days_later(date,lag_price-2)]['close']
            except KeyError:
                try:
                    p_60_day = price_and_fda[ticker].loc[x_days_later(date,lag_price-3)]['close']
                except KeyError:
                    p_60_day = None
    line = (stamp, p_7_day, p_60_day)
    if None not in line:
        prior_and_post_prices.append(line)
print mindate
print maxdate

prior_and_post_prices = pd.DataFrame(prior_and_post_prices)
prior_and_post_prices.columns = ['event', 'close_-7_Day', 'close_+'+str(lag_price)+'_Day']
prior_and_post_prices.head(3)

predictions_and_prices =pd.merge(events_and_predictions, prior_and_post_prices, on='event')

def get_date_from_stamp(stamp):
    return datetime.strptime(stamp.split("/")[1],"%Y-%m-%d")

predictions_and_prices['date'] = predictions_and_prices.event.apply(get_date_from_stamp)

predictions_and_prices['price_%_change'] = ((predictions_and_prices['close_+60_Day']-predictions_and_prices['close_-7_Day']) /  predictions_and_prices['close_-7_Day'])*100

sim_df = predictions_and_prices.sort_values(['date'], axis=0).dropna(axis=0).set_index('date')

sim_df.head(10)

dill.dump(sim_df, open("final_sim_df.pkl", "w"))

sim_df = dill.load(open("final_sim_df.pkl", "r"))

mln_changes = []
rnd_changes = []
all_changes = []
for date in sim_df.iterrows():
    info = date[1]
    if info['pass_prediction'] == 1:
        mln_changes.append(info['price_%_change'])
    else:
        mln_changes.append(0.0)
    if info['pass_random'] == 1:
        rnd_changes.append(info['price_%_change'])
    else:
        rnd_changes.append(0.0)
    if info['pass_all'] == 1:
        all_changes.append(info['price_%_change'])
    else:
        all_changes.append(0.0)

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from math import log
output_notebook()
def calc_changes(start, events):
    prices = [start]
    for event in events:
        last_price = prices[-1]
        prices.append(last_price+(last_price*(event/100)))
    return prices

starting_dollars = [50, 100] #a list of staring values for each investment strategy
p = figure(plot_width=500, 
           plot_height=500, 
           x_axis_label="# of trials traded on",
           y_axis_label="approximate portfolio value"
          )
for start in starting_dollars:
    line_y = calc_changes(start, mln_changes)
    line_x = [x for x in range(len(line_y))]
    p.line(line_x,
           line_y,
           line_width=log(start, 10),
           color = "SteelBlue"
          )
    
    line_y = calc_changes(start, rnd_changes)
    line_x = [x for x in range(len(line_y))]
    p.line(line_x,
           line_y,
           line_width=log(start, 10),
           color = "Tomato"
          )
    
    line_y = calc_changes(start, all_changes)
    line_x = [x for x in range(len(line_y))]
    p.line(line_x,
           line_y,
           line_width=log(start, 10),
           color = "seagreen"
          )
show(p)

