import pyodbc  # Will work if you have PyODBC installed
import pymssql # Will work if you have PyMSSQL installed

from sqlalchemy import create_engine

url = 'mssql+pymssql://steemit:steemit@sql.steemsql.com/DBSteem'

# If you wanted to use ODBC, you would have to use the following URL
# url = 'mssql+pyodbc://steemit:steemit@sql.steemsql.com/DBSteem?driver=SQL Server'

e = create_engine(url)
e.execute("select @@version").fetchone()

import pandas as pd
pd.read_sql("select top 2 * from TxComments", e)

get_ipython().run_cell_magic('time', '', 'q = """\nselect cast(timestamp as date) Day, count(*) as NewUsers\nfrom TxAccountCreates\ngroup by cast(timestamp as date)\norder by Day\n"""\nnew_users = pd.read_sql(q, e, index_col=\'Day\')')

new_users.head(4)

get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set_style()         # Use seaborn-styled plots

new_users.rename(columns={"NewUsers": "New users per day"}).plot(figsize=(8,3));

new_users.cumsum().rename(columns={"NewUsers": "Total user count"}).plot(figsize=(8,3));

get_ipython().run_cell_magic('time', '', 'q = """\nselect cast(timestamp as date) Day, count(*) as NewActiveUsers\nfrom TxAccountCreates\nwhere new_account_name in (select author from TxComments)\ngroup by cast(timestamp as date)\norder by Day\n"""\nnew_active_users = pd.read_sql(q, e, index_col=\'Day\')')

pd.DataFrame({'Total users': new_users.NewUsers.cumsum(),
              'Total active users': new_active_users.NewActiveUsers.cumsum()}).plot(figsize=(8,3));

data = new_active_users[-30:].join(new_users)
data['NewInactiveUsers'] = data.NewUsers - data.NewActiveUsers
data.rename(columns={'NewActiveUsers': 'New active users', 'NewInactiveUsers': 'New inactive users'}, inplace=True)
data[['New active users', 'New inactive users']].plot.bar(stacked=True, figsize=(8,3));

