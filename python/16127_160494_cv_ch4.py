import sys
sys.path.append('../')
import exp
import regression as r

df = exp.get_exp1_data()
df.head()

df.columns

train_cols = ["wp_LST.day", "wp_h", "wp_le", "wp_RNET", "air_temp"]
X, Y = exp.featurize(df, train_cols, ["wp_ch4_gf"])
X, Y, scaler = r.preprocess(X, Y)
X.shape

r.random_forests_cross_val(X, Y, feature_names=train_cols)

r.xgb_trees_cross_val(X, Y, feature_names=train_cols)

r.svc_cross_val(X, Y)

r.dnn_cross_val(X, Y)



