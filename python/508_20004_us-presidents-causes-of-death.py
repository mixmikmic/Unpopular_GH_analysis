get_ipython().magic('load_ext signature')
get_ipython().magic('matplotlib inline')

import requests
import helpers

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ramiro')
chartinfo = 'Author: Ramiro Gómez - ramiro.org • Data: Wikidata - wikidata.org'
infosize = 12

query = '''PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?president ?cause ?dob ?dod WHERE {
    ?pid wdt:P39 wd:Q11696 .
    ?pid wdt:P509 ?cid .
    ?pid wdt:P569 ?dob .
    ?pid wdt:P570 ?dod .
  
    OPTIONAL {
        ?pid rdfs:label ?president filter (lang(?president) = "en") .
    }
    OPTIONAL {
        ?cid rdfs:label ?cause filter (lang(?cause) = "en") .
    }
}'''

url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
data = requests.get(url, params={'query': query, 'format': 'json'}).json()

presidents = []
for item in data['results']['bindings']:
    presidents.append({
        'name': item['president']['value'],
        'cause of death': item['cause']['value'],
        'date of birth': item['dob']['value'],
        'date of death': item['dod']['value']})

df = pd.DataFrame(presidents)
print(len(df))
df.head()

df.dtypes

df['date of birth'] = pd.to_datetime(df['date of birth'])
df['date of death'] = pd.to_datetime(df['date of death'])
df.sort(['date of birth', 'date of death'])

df = df[df['date of birth'] != '1743-04-02']

title = 'US Presidents Causes of Death According to Wikidata'
footer = '''Wikidata lists multiple causes of death for several presidents, that are all included. Thus the total count of causes is
higher than the number of US presidents who died. ''' + chartinfo

df['cause of death'] = df['cause of death'].apply(lambda x: x.capitalize())
s = df.groupby('cause of death').agg('count')['name'].order()

ax = s.plot(kind='barh', figsize=(10, 8), title=title)
ax.yaxis.set_label_text('')
ax.xaxis.set_label_text('Cause of death count')

ax.annotate(footer, xy=(0, -1.16), xycoords='axes fraction', fontsize=infosize)
plt.savefig('img/' + helpers.slug(title), bbox_inches='tight')

get_ipython().magic('signature')

