import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests

req = requests.get('http://www.erobertparker.com/newsearch/vintagechart1.aspx/VintageChart.aspx')

soup = BeautifulSoup(req.text, 'lxml')

charts = soup.find_all(attrs={'class':'chart'})

labels = charts[0].find_all('tr')[1:-1]

label_list = [[(y.find('img'), y.get('rowspan')) 
               for y in x.find_all('td')] 
              for x in labels]

label_len = len(label_list)
empty_list = [np.NaN]*label_len
label_df = pd.DataFrame({'loc1':empty_list, 'loc2':empty_list, 'loc3':empty_list})

for col in range(3):
    pos = 0
    while pos < label_len:

        label = label_list[pos].pop(0)
        
        try:
            text = label[0]
            if text is None:
                text = np.NaN
            else:
                text = text.get('alt')
            
            nrows = label[1]
            if nrows is None:
                nrows = 1
            else:
                nrows = int(nrows)
            
            label_df.loc[pos:pos+nrows, 'loc'+str(col+1)] = text
            pos += nrows
        except:
            pos += 1

label_df.head()

label_df.shape

year_list = [x.text.strip() for x in charts[1].find('tr').find_all('th')]

len(year_list)

ranking_df = pd.DataFrame([[y.text.strip() 
               for y in x.find_all('td')] 
              for x in charts[1].find_all('tr')[1:-1]]).loc[:,1:]

ranking_df.columns = year_list

ranking_df = pd.concat([label_df, ranking_df], axis=1).set_index(['loc1', 'loc2', 'loc3'])

ranking_df = ranking_df.replace('NT',np.NaN).replace('NV',np.NaN)

for col in ranking_df.columns:
    ranking_df[col] = ranking_df[col].str.replace(r"""[A-Z]+""", '')
    ranking_df[col] = ranking_df[col].apply(lambda x: float(x))

ranking_df.head(5)

ranking_df.to_pickle('../priv/pkl/05_vintage_years.pkl')

