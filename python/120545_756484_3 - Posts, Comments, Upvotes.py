get_ipython().magic('matplotlib inline')
import sqlalchemy as sa, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

sns.set_style()
e = sa.create_engine('mssql+pymssql://steemit:steemit@sql.steemsql.com/DBSteem')

def sql(query, index_col="Day"):
    return pd.read_sql(query, e, index_col=index_col)

get_ipython().run_cell_magic('time', '', 'posts = sql("""\nselect \n    cast(timestamp as date) Day,\n    sum(iif(parent_author = \'\', 1, 0)) as Posts,\n    sum(iif(parent_author = \'\', 0, 1)) as Comments\nfrom TxComments\nwhere left(body, 2) <> \'@@\'\ngroup by cast(timestamp as date)\norder by Day\n""")')

posts.plot(figsize=(6,2.5));

df = posts[-30:-1]
df.plot(figsize=(6,2.5))
df.Posts.rolling(7).mean().plot(label="Posts (7-day rolling mean)", ls=":", c="b")
df.Comments.rolling(7).mean().plot(label="Comments (7-day rolling mean)", ls=":", c="g")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4);

get_ipython().run_cell_magic('time', '', 'votes = sql("""\nselect \n    cast(timestamp as date) Day,\n    count(*) as Votes\nfrom TxVotes\ngroup by cast(timestamp as date)\norder by Day\n""")')

votes.plot(figsize=(6,2.5));

df = votes[-30:-1]
df.plot(figsize=(6,2.5), ylim=(0, 500000))
df.Votes.rolling(7).mean().plot(label="Votes (7-day rolling mean)", ls=":", c="b")
plt.legend(loc='lower left');

get_ipython().run_cell_magic('time', '', 'sql("""\nselect\n    cast(expiration as date) Day,\n    count(*) as Transactions\nfrom Transactions\ngroup by cast(expiration as date)\norder by Day\n""").plot(figsize=(6,2.5));')

