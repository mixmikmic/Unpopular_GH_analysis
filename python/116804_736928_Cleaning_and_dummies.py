get_ipython().magic('matplotlib inline')
import pickle
get_ipython().magic('run helper_functions.py')
pd.options.display.max_columns = 1000
plt.rcParams["figure.figsize"] = (15,10)
from datetime import datetime
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = unpickle_object("non_current_df.pkl") #loans that are 'complete'

df.shape

df['loan_status'].unique()

mask = df['loan_status'] != "Fully Paid"
rows_to_change = df[mask]
rows_to_change.loc[:, 'loan_status'] = 'Late'
df.update(rows_to_change)

df['loan_status'].unique() #sweet!

df.shape # no dimensionality lost

plot_corr_matrix(df)

no_desc = []
for column in df.columns:
    try:
        print(column+":",lookup_description(column)," DataType:", df[column].dtype)
        print()
    except KeyError:
        no_desc.append(column)

columns_to_drop = ["id", "member_id", "emp_title","desc","title","out_prncp","out_prncp_inv","total_pymnt","total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee","last_pymnt_d", "last_pymnt_amnt","next_pymnt_d", "last_credit_pull_d", "collections_12_mths_ex_med","mths_since_last_major_derog", "all_util", ]

# df.loc[:, ["loan_amnt","funded_amnt","out_prncp","out_prncp_inv","total_pymnt","total_pymnt_inv","total_rec_prncp","last_credit_pull_d"]]

no_desc

df['verification_status_joint'].unique()

df['total_rev_hi_lim'].unique()

df['verification_status_joint'].dtype

df['total_rev_hi_lim'].dtype

df.drop(columns_to_drop, axis=1, inplace=True)

df.shape #just what we expected

df["policy_code"] = df["policy_code"].astype('object')

df['pct_tl_nvr_dlq'] = df['pct_tl_nvr_dlq'].apply(lambda x: x/100)
df['percent_bc_gt_75'] = df['percent_bc_gt_75'].apply(lambda x: x/100)

object_columns = df.select_dtypes(include=['object']).columns

for c in object_columns:
    df.loc[df[df[c].isnull()].index, c] = "missing"

obj_df = df.select_dtypes(include=['object'])

obj_df_cols = obj_df.columns

for col in obj_df_cols:
    df[col] = df[col].astype("category")
    
df.dtypes.unique() #This is what we wanted!

df.shape

df.head()

unique_val_dict = {}
for col in df.columns:
    if col not in unique_val_dict:
        unique_val_dict[col] = df[col].unique()

unique_val_dict #will use this later when making flask app.

category_columns = df.select_dtypes(include=['category']).columns
df = pd.get_dummies(df, columns=category_columns, drop_first=True)

df.shape

float_columns = df.select_dtypes(include=['float64']).columns

for c in float_columns:
    df.loc[df[df[c].isnull()].index, c] = np.nan

pickle_object(unique_val_dict, "unique_values_for_columns")

pickle_object(df, "dummied_dataset")

df = unpickle_object("dummied_dataset.pkl")

df.head()



