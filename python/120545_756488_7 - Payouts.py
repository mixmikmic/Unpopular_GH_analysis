get_ipython().magic('matplotlib inline')
import sqlalchemy as sa, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

sns.set_style()
e = sa.create_engine('mssql+pymssql://steemit:steemit@sql.steemsql.com/DBSteem')

def sql(query, index_col=None):
    return pd.read_sql(query, e, index_col=index_col)

sql("""select top 5 
            author, 
            permlink,
            pending_payout_value,
            total_pending_payout_value,
            total_payout_value,
            curator_payout_value
        from Comments""")

get_ipython().run_cell_magic('time', '', 'max_daily_payout = sql("""\nselect \n    cast(created as date) as Date,\n    max(pending_payout_value \n        + total_pending_payout_value\n        + total_payout_value\n        + curator_payout_value) as Payout\nfrom Comments\ngroup by cast(created as date)\norder by Date\n""", "Date")')

max_daily_payout.plot(title="Highest daily payout (SBD)")
max_daily_payout[-30:].plot(title="Highest daily payout (last 30 days - SBD)", 
                            ylim=(0,1500));

superpost = sql("""
    select * from Comments 
    where pending_payout_value 
        + total_pending_payout_value
        + total_payout_value
        + curator_payout_value > 45000""")

superpost[['author', 'category', 'permlink']]

get_ipython().run_cell_magic('time', '', 'avg_payouts = sql("""\nwith TotalPayouts as (\n    select \n        cast(created as date) as [Date],\n        iif(parent_author = \'\', 1, 0) as IsPost,\n        pending_payout_value \n            + total_pending_payout_value\n            + total_payout_value\n            + curator_payout_value as Payout\n    from Comments\n)\nselect\n    Date,\n    IsPost,\n    avg(Payout) as Payout,\n    count(*)    as Number\nfrom TotalPayouts\ngroup by Date, IsPost\norder by Date, IsPost\n""", "Date")')

posts = avg_payouts[avg_payouts.IsPost == 1][-30:]
comments = avg_payouts[avg_payouts.IsPost == 0][-30:]

fig, ax = plt.subplots()

# Plot payouts using left y-axis
posts.Payout.plot(ax=ax, c='r', label='Average payout (Posts)')
comments.Payout.plot(ax=ax, c='r', ls=':', label='Average payout (Comments)')
ax.set_ylabel('Payout')
ax.legend(loc='center left')

# Plot post counts using right y-axis
ax2 = ax.twinx()
posts.Number.plot(ax=ax2, c='b', label='Count (Posts)')
comments.Number.plot(ax=ax2, c='b', ls=':', label='Count (Comments)', 
                     ylim=(0,90000))
ax2.set_ylabel('Count')
ax2.legend(loc='center right')
ax2.grid(ls='--', c='#9999bb')

get_ipython().run_cell_magic('time', '', 'median_payout = sql("""\nwith TotalPayouts as (\n    select \n        cast(created as date) as [Date],\n        pending_payout_value \n            + total_pending_payout_value\n            + total_payout_value\n            + curator_payout_value as Payout\n    from Comments\n    where parent_author = \'\'\n)\nselect\n    distinct Date,\n    percentile_cont(0.5) \n        within group(order by Payout) \n        over(partition by Date) as [Median Payout]\nfrom TotalPayouts\norder by Date\n""", "Date")')

df = median_payout[-30:]
df.plot(ylim=(0,0.1))
df['Median Payout'].rolling(7).mean().plot(
                   label='Median Payout (7-day avg)', ls=':', c='b')
plt.legend();

