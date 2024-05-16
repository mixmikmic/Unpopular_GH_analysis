get_ipython().magic('matplotlib inline')
import sys
sys.path.append('../')
import exp
import regression as r
import numpy as np

df = exp.get_exp1_data()
df.head()

X_cols, Y_cols = ["wp_LST.day", "wp_h", "wp_le", "net_rad", "avg_air_temp"], ["wp_er"]
train_years, test_years = [2013], [2014, 2015]
X_train, Y_train = exp.featurize(df, X_cols, Y_cols, years=train_years)
X_test, Y_test = exp.featurize(df, X_cols, Y_cols, years=test_years)

Y_pred = r.predict(r.random_forests(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)

Y_pred = r.predict(r.xgb_trees(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)

Y_pred = r.predict(r.svm(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)

Y_pred = r.predict(r.dnn(), X_train, Y_train, X_test, Y_test)
r.visualize_preds(df, Y_test, Y_pred, test_years=test_years)

