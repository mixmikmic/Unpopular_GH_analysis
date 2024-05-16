import requests
import pandas as pd
import time

# Used to loop over countries 5 at a time.
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

c_codes = pd.read_csv('codes/country_codes.csv').set_index('id')

prod_type = 'C'  # Commodity
freq = 'A'       # Annual 
classification = 'HS' # harmonized system
prod = '440710'   # HS 6-digit production ID
years = ['2005', '2010', '2015']
base = 'http://comtrade.un.org/api/get?'
url = '{}max=50000&type={}&freq={}&px={}'.format(
    base, prod_type, freq, classification)
df = pd.DataFrame(columns=['period', 'pt3ISO', 'rt3ISO', 'TradeValue'])

for n in chunker(c_codes.index.values[1:], 5):
    req = '&ps={}&r=all&p={}&rg=2&cc={}'.format(
        '%2C'.join(years), '%2C'.join(n), prod)
    r = requests.get('{}{}'.format(url, req)).json()['dataset']
    for f in r:
        df = df.append([f['period'], f['pt3ISO'], f['rt3ISO'], f['TradeValue']])
    time.sleep(5)

df.fillna(value='TWN', inplace=True)
df.head()

df.to_csv('440710.csv')

