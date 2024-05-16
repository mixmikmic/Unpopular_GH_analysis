from fbprophet import Prophet
from sklearn.metrics import r2_score
get_ipython().magic('run helper_functions.py')
get_ipython().magic('autosave 120')
get_ipython().magic('matplotlib inline')
get_ipython().magic('run prophet_helper.py')
get_ipython().magic('run prophet_baseline_btc.py')
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (15,10)
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['legend.fontsize'] = 20
plt.style.use('fivethirtyeight')
pd.set_option('display.max_colwidth', -1)
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = unpickle_object("blockchain_info_df.pkl")
df.head()

df_btc = pd.DataFrame(df['mkt_price'])

true, pred = prophet_baseline_BTC(df_btc, 30, "mkt_price")

r2_score(true, pred) #we see that our baseline model just predicts 44% of the variation when predicting price

plt.plot(pred)
plt.plot(true)
plt.legend(["Prediction", 'Actual'], loc='upper left')
plt.xlabel("Prediction #")
plt.ylabel("Price")
plt.title("TS FB Prophet Baseline - Price Prediction");

df_btc_pct = df_btc.pct_change()
df_btc_pct.rename(columns={"mkt_price": "percent_change"}, inplace=True)
df_btc_pct = df_btc_pct.iloc[1:, :]
print(df_btc_pct.shape)
df_btc_pct.head()

true_pct, pred_pct = prophet_baseline_BTC(df_btc_pct, 30, "percent_change")

r2_score(true_pct, pred_pct)

plt.plot(pred_pct)
plt.plot(true_pct)
plt.legend(["Prediction", 'Actual'], loc='upper left')
plt.xlabel("Prediction #")
plt.ylabel("Price")
plt.title("TS FB Prophet Baseline - Price Prediction");

prices_to_be_multiplied = df.loc[pd.date_range(start="2017-01-23", end="2017-02-21"), "mkt_price"]

forecast_price_lst = []
for index, price in enumerate(prices_to_be_multiplied):
    predicted_percent_change = 1+float(pred_pct[index])
    forecasted_price = (predicted_percent_change)*price
    forecast_price_lst.append(forecasted_price)

ground_truth_prices = df.loc[pd.date_range(start="2017-01-24", end="2017-02-22"), "mkt_price"]
ground_truth_prices = list(ground_truth_prices)

r2_score(ground_truth_prices, forecast_price_lst) # such an incredible result! This is what we have to beat with my nested TS model

plt.plot(forecast_price_lst)
plt.plot(ground_truth_prices)
plt.legend(["Prediction", 'Actual'], loc='upper left')
plt.xlabel("Prediction #")
plt.ylabel("Price")
plt.title("TS FB Prophet Baseline - Price Prediction");



