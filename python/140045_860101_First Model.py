import dill
import numpy as np

data = dill.load(open("annotated_traning_data.pkl", "r"))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

decisions = np.array([x[2] for x in data])
prices = [x[3][0] for x in data]

price_array = np.array(prices).astype(float)

fixed_prices = np.nan_to_num(price_array)

train_prices, test_prices, train_decisions, test_decisions = train_test_split(fixed_prices, decisions, test_size = .3)

basic_classifier = LogisticRegression()

basic_classifier.fit(train_prices,train_decisions)

basic_classifier.predict(test_prices)

results = cross_val_score(basic_classifier, test_prices, test_decisions, cv = 5)

results

