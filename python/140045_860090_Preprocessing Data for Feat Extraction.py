import dill
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')

data = dill.load(open('normalized_stock_price_slices.pkl', 'r'))

def assemble_frame(datum):
    df = pd.DataFrame(datum[3][0], columns=['date','norm_price'])
    df['event'] = datum[0]+"/"+datum[1]
    df['outcome'] = int(datum[2])
    return df

first = True

for line in tqdm_notebook(data):
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

agg_data.head(2)

agg_data['null'] = pd.isnull(agg_data).apply(lambda x: sum(x) , axis=1)

agg_data

agg_data['null'] = pd.isnull(agg_data).apply(lambda x: sum(x) , axis=1)

cleaned_agg = agg_data[agg_data['null'] == 0]

cleaned_agg

dill.dump(cleaned_agg, open("unified_and_stamped_dataframe.pkl", "w"))

from sklearn.cross_validation import train_test_split

train_data, test_data = train_test_split(data, train_size = .8)

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

dill.dump(train_clean, open("train_df.pkl", "w"))
dill.dump(test_clean, open("test_df.pkl", "w"))

