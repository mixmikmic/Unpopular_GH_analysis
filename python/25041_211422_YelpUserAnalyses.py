get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from pandas.io.json import json_normalize

# Written by Reddit user "ryptophan"
# read the entire file into a python array
with open('yelp_academic_dataset_user.json', 'r') as f:
    data = f.readlines()

# remove the trailing "\n" from each line
data = list(map(lambda x: str(x).rstrip(), data))

# each element of 'data' is an individual JSON object.
# i want to convert it into an *array* of JSON objects
# which, in and of itself, is one large JSON object
# basically... add square brackets to the beginning
# and end, and have all the individual business JSON objects
# separated by a comma
data_json_str = "[" + ','.join(data) + "]"

# now, load it into pandas
user_df = pd.read_json(data_json_str)

plt.figure()
plt.hist(user_df.fans, 100);
plt.title('Number of fans histogram')

plt.figure()
plt.hist(user_df.review_count, 100);
plt.title('Review count histogram')

plt.figure()
plt.hist(user_df.average_stars, 100);
plt.title('Average stars histogram')

user_df_clean = user_df.ix[(user_df.fans > 10),:]
user_df_clean = user_df_clean.ix[(user_df_clean.review_count > 25),:]
user_df_clean = user_df_clean.ix[(user_df_clean.review_count < 1500),:]
print(user_df_clean.shape)

plt.figure()
plt.hist(user_df_clean.fans, 100);
plt.title('Number of fans histogram')

plt.figure()
plt.hist(user_df_clean.review_count, 100);
plt.title('Review count histogram')

plt.figure()
plt.hist(user_df_clean.average_stars, 40);
plt.title('Average stars histogram')

plt.scatter(user_df_clean.average_stars, user_df_clean.fans);
plt.xlabel('average stars')
plt.ylabel('number of fans');



