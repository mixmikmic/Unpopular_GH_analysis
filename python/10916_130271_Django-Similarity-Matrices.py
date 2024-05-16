import re
import pandas as pd
pd.options.display.max_columns = 9
pd.options.display.max_rows = 3
np = pd.np
np.norm = np.linalg.norm
from datetime import datetime, date
import json
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
from sklearn.feature_extraction.text import TfidfVectorizer  # equivalent to TFIDFTransformer(CountVectorizer())
from django.db.models import Sum
from pacs.models import CampaignDetail, WorkingTransactions
import django
django.setup()
CampaignDetail.objects.count(), WorkingTransactions.objects.count()

qs = CampaignDetail.objects.annotate(net_amount=Sum('workingtransactions__amount')).values().all()
print('Net transactions: {:,}M'.format(round(sum(qs.values_list('net_amount', flat=True)) / 1e6)))

df = pd.DataFrame.from_records(qs)
df.columns

df = df[df.committee_name.astype(bool)].copy()
df

qs_pos = CampaignDetail.objects.filter(committee_name__isnull=False, workingtransactions__amount__gt=0)
qs_pos = qs_pos.annotate(pos_amount=Sum('workingtransactions__amount'))
df_pos = df.join(pd.DataFrame.from_records(qs_pos.values('pos_amount').all())['pos_amount'])
df_pos

df = pd.DataFrame.from_records(qs)
df = pd.DataFrame(df[df.committee_name.astype(bool)])
df['pos_amount'] = pd.DataFrame.from_records(qs_pos.values('pos_amount').all())['pos_amount']
df

df == df_pos

pd.options.display.max_rows = 6
(df == df_pos).mean()

(df == df_pos).mean() + df.isnull().mean()

qs_neg = CampaignDetail.objects.filter(workingtransactions__amount__lt=0)
qs_neg = qs_neg.annotate(neg_amount=Sum('workingtransactions__amount'))
df = df.join(pd.DataFrame.from_records(qs_neg.values('neg_amount').all())['neg_amount'])
df

print('Positve transactions: {:,} M'.format(round(sum(qs_pos.values_list('pos_amount', flat=True)) / 1e6)))

print('Negative transactions: {:,} M'.format(round(sum(qs_neg.values_list('neg_amount', flat=True)) / 1.e6, 2)))

print('Net net transactions: {:,} M'.format(round(sum(qs.values_list('net_amount', flat=True)) / 1.e6)))

df.sum()

print('Net amount: ${:} M'.format(round(df.sum()[['pos_amount', 'neg_amount']].sum()/1e6, 2)))

print('Volume: ${:} M'.format(round(np.abs(df.sum()[['pos_amount', 'neg_amount']]).sum()/1e6, 2)))

filer_id = set(pd.DataFrame.from_records(WorkingTransactions.objects.values(
               'filer_id').all()).dropna().values.T[0])
payee_id = set(pd.DataFrame.from_records(WorkingTransactions.objects.values(
               'contributor_payee_committee_id').all()).dropna().values.T[0])
com_id = set()
len(payee_id.intersection(filer_id)) * 1. / len(filer_id)

qs = WorkingTransactions.objects.filter(filer_id__isnull=False, 
                                        contributor_payee_committee_id__isnull=False,
                                        amount__gt=0)
df_trans = pd.DataFrame.from_records(qs.values().all())
_, trans = df_trans.iterrows().next()
print(trans)
print(trans.index.values)

ids = [int(i) for i in payee_id.intersection(filer_id)]
id_set = set(ids)
id_str = [str(int(i)) for i in ids]
N = len(ids)
cov = pd.DataFrame(np.zeros((N, N)),
                   index=pd.Index(id_str, name='payee'),
                   columns=pd.Index(id_str, name='filer'))
print(cov)
for rownum, trans in df_trans.iterrows():
    fid = trans['filer_id_id']
    # print(trans.index.values)
    cid = trans['contributor_payee_committee_id']
    if fid in id_set and cid in id_set:
#         if not (fid % 100):
#             print(cov[str(fid)][str(cid)])
        #only populate the upper
        if fid > cid:
            fid, cid = cid, fid
        amount = abs(trans['amount'])
        if amount > 0:
            cov[str(fid)][str(cid)] += amount
cov.describe()
    

cov

cov.sum()

