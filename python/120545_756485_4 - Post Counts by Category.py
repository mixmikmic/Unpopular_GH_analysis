get_ipython().magic('matplotlib inline')
import sqlalchemy as sa, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

sns.set_style()
e = sa.create_engine('mssql+pymssql://steemit:steemit@sql.steemsql.com/DBSteem')

def sql(query, index_col=None):
    return pd.read_sql(query, e, index_col=index_col)

get_ipython().run_cell_magic('time', '', 'top_categories = sql("""\nselect top 20\n    parent_permlink as Category,\n    count(*) as Count\nfrom TxComments\nwhere\n    parent_author = \'\'\n    and left(body, 2) <> \'@@\'\ngroup by parent_permlink\norder by Count desc\n""", "Category")')

ax = top_categories.plot.bar(figsize=(7,3), ylim=(0,200000));
for i,(k,v) in enumerate(top_categories.itertuples()):
    ax.annotate(v, xy=(i, v+25000), ha='center', rotation=45, fontsize=8)

get_ipython().run_cell_magic('time', '', 'top_day_categories = sql("""\nselect top 20\n    parent_permlink as Category,\n    count(*) as Count\nfrom TxComments\nwhere\n    parent_author = \'\'\n    and left(body, 2) <> \'@@\'\n    and cast(timestamp as date) = \'2017-08-10\'    -- This line is new\ngroup by parent_permlink\norder by Count desc\n""", "Category")')

ax = top_day_categories.plot.bar(figsize=(7,3), ylim=(0,1500));
for i,(k,v) in enumerate(top_day_categories.itertuples()):
    ax.annotate(v, xy=(i, v+50), ha='center', fontsize=8)

