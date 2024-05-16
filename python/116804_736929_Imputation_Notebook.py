get_ipython().magic('matplotlib inline')
import pickle
get_ipython().magic('run helper_loans.py')
pd.options.display.max_columns = 1000
plt.rcParams["figure.figsize"] = (15,10)
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

df = unpickle_object("dummied_dataset.pkl")

df.shape

#this logic will be important for flask data entry.

float_columns = df.select_dtypes(include=['float64']).columns

for col in float_columns:
    if "mths" not in col:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        if col == "inq_last_6mths":
            df[col].fillna(0, inplace=True)
        elif col == "mths_since_last_delinq":
            df[col].fillna(999, inplace=True)
        elif col == "mths_since_last_record":
            df[col].fillna(999, inplace=True)
        elif col == "collections_12_mths_ex_med":
            df[col].fillna(0, inplace=True)
        elif col == "mths_since_last_major_derog":
            df[col].fillna(999, inplace=True)
        elif col == "mths_since_rcnt_il":
            df[col].fillna(999, inplace=True)
        elif col == "acc_open_past_24mths":
            df[col].fillna(0, inplace=True)
        elif col == "chargeoff_within_12_mths":
            df[col].fillna(0, inplace=True)
        elif col == "mths_since_recent_bc":
            df[col].fillna(999, inplace=True)
        elif col == "mths_since_recent_bc_dlq":
            df[col].fillna(999, inplace=True)
        elif col == "mths_since_recent_inq":
            df[col].fillna(999, inplace=True)
        elif col == "mths_since_recent_revol_delinq":
            df[col].fillna(999, inplace=True)

scaler = StandardScaler()
matrix_df = df.as_matrix()
matrix = scaler.fit_transform(matrix_df)
scaled_df = pd.DataFrame(matrix, columns=df.columns)

scaled_df.shape

pickle_object(df, "CLASSIFICATION DF")

pickle_object(scaled_df, "GLM DATAFRAME")

#legacy code - how I would implement a random forest imputation

# good_features = df[df['mths_since_last_record'].notnull()]
# good_values = good_features.drop(['mths_since_last_record', 'loan_status_Late'], axis=1).values
# good_indicies = good_features.index
# good_target = df.loc[good_indicies, :]['mths_since_last_record'].values
# to_predict_array = df[df['mths_since_last_record'].isnull()].drop(['mths_since_last_record', 'loan_status_Late'], axis=1).values
# to_prediact_index = df[df['mths_since_last_record'].isnull()].index

# model = RandomForestClassifier(n_estimators=25,criterion='entropy', n_jobs=-1)

# model.fit(good_values, good_target)

# impute_values = model.predict(to_predict_array)

# # df.loc[to_predict_index, 'mths_since_last_record'] = impute_values



