import json
import pandas as pd
from pandas.io.json import json_normalize

with open("blue-bottle-coffee-san-francisco.json", "r") as f:
     data = json.load(f)

df = json_normalize(data, "reviewList")
df.head(3)

df.dtypes

df["ratings"] = pd.to_numeric(df["ratings"])
df.dtypes

df["ratings"].describe()

project_id = "your-project-ID"
df.to_gbq("mydataset.mytable", project_id=project_id, verbose=True, if_exists="replace")

query = "SELECT * FROM mydataset.mytable LIMIT 5"
pd.read_gbq(query=query, dialect="standard", project_id=project_id)

