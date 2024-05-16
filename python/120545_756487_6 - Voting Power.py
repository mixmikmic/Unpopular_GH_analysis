get_ipython().magic('matplotlib inline')
import sqlalchemy as sa, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

sns.set_style()
e = sa.create_engine('mssql+pymssql://steemit:steemit@sql.steemsql.com/DBSteem')

def sql(query, index_col=None):
    return pd.read_sql(query, e, index_col=index_col)

sql("select name, vesting_shares from Accounts where name = 'konstantint'")

sql("select count(*) from Accounts where right(vesting_shares, 6) != ' VESTS'")

sql("select top 3 name, cast(left(vesting_shares, len(vesting_shares)-6) as float) from Accounts")

sql("""
select 
    name, 
    cast(left(vesting_shares, len(vesting_shares)-6) as float)*484.529/1000000 as SP
from Accounts
where name = 'konstantint'""")

sql("""
select count(*) as DeadFishes
from Accounts 
where 
  last_post < dateadd(day, -7, cast(getdate() as date))
  and last_vote_time < dateadd(day, -7, cast(getdate() as date))
""")

levels = sql("""
declare @weekAgo date = dateadd(day, -7, cast(getdate() as date));

with Data as (
    select 
        iif(last_post < @weekAgo and last_vote_time < @weekAgo, 1, 0) as Dead,
        cast(left(vesting_shares, len(vesting_shares)-6) as float) as Vests
    from Accounts),

Levels as (
    select 
        case when Dead = 1 then 0
             when Vests < 1000000 then 1
             when Vests < 10000000 then 2
             when Vests < 100000000 then 3
             when Vests < 1000000000 then 4
             else 5
             end as Level,
         Vests
     from Data)

select Level, count(*) as Count from Levels
group by Level order by Level
""", "Level")

levels

levels.plot.bar();

f0 = "http://steemitboard.com/@initminer/level.png"
f1 = "http://steemitboard.com/@konstantint/level.png"
f2 = "http://steemitboard.com/@rycharde/level.png"
f3 = "http://steemitboard.com/@arcange/level.png"
f4 = "http://steemitboard.com/@teamsteem/level.png"
f5 = "http://steemitboard.com/@dan/level.png"
urls = [f0, f1, f2, f3, f4, f5]

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage import io

imgs = [io.imread(url) for url in urls]
ax = levels.plot.bar(figsize=(8, 4), ylim=(0, 280000))
plt.xlabel("Level", labelpad=40)
plt.xticks()

for i,v in enumerate(levels.Count):
    ax.annotate(v, xy=(i, v+5000), ha='center')
    oi = OffsetImage(imgs[i], zoom = 0.15)
    box = AnnotationBbox(oi, (i, 0), frameon=False, box_alignment=(0.5, 1))
    ax.add_artist(box)

total_power = sql("""
declare @weekAgo date = dateadd(day, -7, cast(getdate() as date));

with Data as (
    select 
        iif(last_post < @weekAgo and last_vote_time < @weekAgo, 1, 0) as Dead,
        cast(left(vesting_shares, len(vesting_shares)-6) as float) as Vests
    from Accounts),

Levels as (
    select 
        case when Dead = 1 then 0
             when Vests < 1000000 then 1
             when Vests < 10000000 then 2
             when Vests < 100000000 then 3
             when Vests < 1000000000 then 4
             else 5
             end as Level,
         Vests
     from Data)

-- The line below was modified
select Level, sum(Vests)/1000000 as [Total Power] from Levels
group by Level order by Level
""", "Level")

ax = total_power.plot.bar(figsize=(8, 4), ylim=(0, 300000))
plt.xlabel("Level", labelpad=40)
plt.xticks()

for i,v in enumerate(total_power['Total Power']):
    ax.annotate(int(round(v)), xy=(i, v+5000), ha='center')
    oi = OffsetImage(imgs[i], zoom = 0.15)
    box = AnnotationBbox(oi, (i, 0), frameon=False, box_alignment=(0.5, 1))
    ax.add_artist(box)

total_power2 = sql("""
declare @weekAgo date = dateadd(day, -7, cast(getdate() as date));

with Data as (
    select 
        iif(last_post < @weekAgo and last_vote_time < @weekAgo, 1, 0) as Dead,
        cast(left(vesting_shares, len(vesting_shares)-6) as float) as Vests
    from Accounts),

Levels as (
    select 
        case when Vests < 1000000 then 1   -- The previous line removed
             when Vests < 10000000 then 2
             when Vests < 100000000 then 3
             when Vests < 1000000000 then 4
             else 5
             end as Level,
         Vests
     from Data)

select Level, sum(Vests)/1000000 as [Total Power] from Levels
group by Level order by Level
""", "Level")

ax = total_power2.plot.bar(figsize=(8, 4), ylim=(0, 350000))
plt.xlabel("Level", labelpad=40)
plt.xticks()

for i,v in enumerate(total_power2['Total Power']):
    ax.annotate(int(round(v)), xy=(i, v+5000), ha='center')
    oi = OffsetImage(imgs[i+1], zoom = 0.15)
    box = AnnotationBbox(oi, (i, 0), frameon=False, box_alignment=(0.5, 1))
    ax.add_artist(box)

