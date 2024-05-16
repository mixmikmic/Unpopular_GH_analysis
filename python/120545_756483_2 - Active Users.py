get_ipython().magic('matplotlib inline')
from sqlalchemy import create_engine
import pandas as pd
import seaborn as sns
sns.set_style()

url = 'mssql+pymssql://steemit:steemit@sql.steemsql.com/DBSteem'
e = create_engine(url)

get_ipython().run_cell_magic('time', '', 'q = """\nselect \n    cast(timestamp as date) Day,\n    count(distinct author) as [Active authors]\nfrom TxComments\ngroup by cast(timestamp as date)\norder by Day\n"""\nactive_authors = pd.read_sql(q, e, index_col=\'Day\')')

get_ipython().run_cell_magic('time', '', 'q = """\nselect \n    cast(timestamp as date) Day,\n    count(distinct voter) as [Active voters]\nfrom TxVotes\ngroup by cast(timestamp as date)\norder by Day\n"""\nactive_voters = pd.read_sql(q, e, index_col=\'Day\')')

get_ipython().run_cell_magic('time', '', 'q = """\nselect \n    cast(timestamp as date) Day,\n    count(distinct name) as [Active users]\nfrom (select timestamp, voter as name from TxVotes \n      union \n      select timestamp, author as name from TxComments) data\ngroup by cast(timestamp as date)\norder by Day\n"""\nactive_users = pd.read_sql(q, e, index_col=\'Day\')')

df = active_users.join(active_voters).join(active_authors)
df.plot(figsize=(8,3))
df[-30:].plot(figsize=(8,3), ylim=(0, 25000));

get_ipython().run_cell_magic('time', '', 'q = """\nselect \n    year(timestamp) Year,\n    month(timestamp) Month,\n    count(distinct author) as [Active authors]\nfrom TxComments\ngroup by year(timestamp), month(timestamp)\norder by Year, Month\n"""\nactive_monthly_authors = pd.read_sql(q, e, index_col=[\'Year\', \'Month\'])')

get_ipython().run_cell_magic('time', '', 'q = """\nselect \n    year(timestamp) Year,\n    month(timestamp) Month,\n    count(distinct voter) as [Active voters]\nfrom TxVotes\ngroup by year(timestamp), month(timestamp)\norder by Year, Month\n"""\nactive_monthly_voters = pd.read_sql(q, e, index_col=[\'Year\', \'Month\'])')

get_ipython().run_cell_magic('time', '', 'q = """\nselect \n    year(timestamp) Year,\n    month(timestamp) Month,\n    count(distinct name) as [Active users]\nfrom (select timestamp, voter as name from TxVotes \n      union \n      select timestamp, author as name from TxComments) data\ngroup by year(timestamp), month(timestamp)\norder by Year, Month\n"""\nactive_monthly_users = pd.read_sql(q, e, index_col=[\'Year\', \'Month\'])')

df = active_monthly_users.join(active_monthly_voters).join(active_monthly_authors)
df.plot(figsize=(8,3));

ax = df.plot(alpha=0.8,figsize=(8,3));
au = df['Active users']
au.plot(style='.')
for i in range(len(df)):
    ax.annotate(au[i], xy=(i, au[i]+1800), ha='center', fontsize=8)

