import numpy as np
import pandas as pd
import dill
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

test_df = dill.load(open("test_df.pkl", "r"))
train_df = dill.load(open("train_df.pkl", "r"))

train_df

train_feats = extract_features(train_df[['norm_price', 'event_stamp', 'date_stamp']], 
                              column_id="event_stamp", column_sort="date_stamp", 
                              column_value="norm_price", n_jobs=0)

train_feats.dropna(axis=1)

arr = train_feats.isnull().values
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if arr[i,j]:
            print(i,j,arr[i,j])

from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features

y = train_df[['event_stamp', 'outcome']]

groups = y.groupby('event_stamp')

outcomes = groups.head(1).set_index('event_stamp')['outcome']

features_filtered_direct =extract_relevant_features(train_df[['norm_price', 'event_stamp', 'date_stamp']], 
                                                    outcomes, 
                                                    column_id="event_stamp", 
                                                    column_sort="date_stamp", 
                                                    column_value="norm_price",
                                                    n_jobs=0)

[col for col in features_filtered_direct.columns]

from sklearn.ensemble import RandomForestClassifier

pharma_forest = RandomForestClassifier(max_features= 15)

pharma_forest.fit(features_filtered_direct, outcomes)

from sklearn.model_selection import cross_val_score

test_y = test_df[['event_stamp', 'outcome']]
groups_test = test_y.groupby('event_stamp')
outcomes_test = groups.head(1).set_index('event_stamp')['outcome']

test_feats =extract_features(test_df[['norm_price', 'event_stamp', 'date_stamp']],  
                             column_id="event_stamp", 
                             column_sort="date_stamp", 
                             column_value="norm_price", 
                             n_jobs=0)[features_filtered_direct.columns]

test_feats

cross_val_score(estimator=pharma_forest, X=test_feats, y=outcomes_test, cv = 100)



