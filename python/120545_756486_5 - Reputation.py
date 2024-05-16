get_ipython().magic('matplotlib inline')
import sqlalchemy as sa, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

sns.set_style()
e = sa.create_engine('mssql+pymssql://steemit:steemit@sql.steemsql.com/DBSteem')

def sql(query, index_col=None):
    return pd.read_sql(query, e, index_col=index_col)

sql("select * from information_schema.tables")

sql("select top 3 * from Accounts")

sql("""
select 
    name, 
    reputation as raw_reputation,
    cast(log10(reputation)*9 - 56 as int) as reputation
from Accounts 
where name = 'konstantint'""")

get_ipython().run_cell_magic('time', '', 'reputations = sql("""\nwith Data as \n    (select \n       cast(log10(isnull(reputation, 0))*9 - 56 as int) as Reputation\n     from Accounts\n     where reputation > 0)\n\nselect \n    Reputation, count(*) as Count\nfrom Data \ngroup by Reputation\nhaving Reputation > 25\norder by Reputation desc""", "Reputation")')

reputations.plot.bar(figsize=(8, 4));

