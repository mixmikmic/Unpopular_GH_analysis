get_ipython().magic('matplotlib inline')
import pickle
get_ipython().magic('run helper_functions.py')
get_ipython().magic('run s3.py')
pd.options.display.max_columns = 1000
plt.rcParams["figure.figsize"] = (15,10)
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import psycopg2
import os

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

top_35 = ["int_rate", 
          "dti", 
          "term_ 60 months",
          "bc_open_to_buy",
          "revol_util",
          "installment",
          "avg_cur_bal",
          "tot_hi_cred_lim",
          "revol_bal",
          "funded_amnt_inv",
          "bc_util",
          "tot_cur_bal",
          "total_bc_limit",
          "total_rev_hi_lim",
          "funded_amnt",
          "loan_amnt",
          "mo_sin_old_rev_tl_op",
          "total_bal_ex_mort",
          "issue_d_Dec-2016",
          "total_acc",
          "mo_sin_old_il_acct",
          "mths_since_recent_bc",
          "total_il_high_credit_limit",
          "inq_last_6mths",
          "acc_open_past_24mths",
          "mo_sin_rcnt_tl",
          "mo_sin_rcnt_rev_tl_op",
          "percent_bc_gt_75",
          "num_rev_accts",
          "mths_since_last_delinq",
          "open_acc",
          "mths_since_recent_inq",
          "grade_B",
          "num_bc_tl",
          "loan_status_Late"]

df_reduced_features = df.loc[:, top_35]

df_reduced_features.shape

scaler = StandardScaler()
matrix_df = df_reduced_features.as_matrix()
matrix = scaler.fit_transform(matrix_df)
scaled_df = pd.DataFrame(matrix, columns=df_reduced_features.columns)

scaler = StandardScaler()
matrix_df = df_reduced_features.as_matrix()
scalar_object_35 = scaler.fit(matrix_df)
matrix = scalar_object_35.transform(matrix_df)
scaled_df_35 = pd.DataFrame(matrix, columns=df_reduced_features.columns)

check = scaled_df_35 == scaled_df # lets pickle the scaler

check.head()

pickle_object(scalar_object_35, "scaler_35_features")

pickle_object(scaled_df, "rf_df_35")

upload_to_bucket('rf_df_35.pkl', "rf_df_35.pkl","gabr-project-3")

upload_to_bucket("scaler_35_features.pkl", "scaler_35_features.pkl", "gabr-project-3")

df = unpickle_object("rf_df_35.pkl")

engine = create_engine(os.environ["PSQL_CONN"])

df.to_sql("dummied_dataset", con=engine)

pd.read_sql_query('''SELECT * FROM dummied_dataset LIMIT 5''', engine)



