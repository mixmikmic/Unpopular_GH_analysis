NumberOfArticles = 10000
ArticleMinLength = 1000

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
#from lxml import objectify

# takes a while
tree = ET.parse('simplewiki-20160701-pages-articles-multistream.xml') 

root = tree.getroot()
#print(root)
#print(root.attrib)
#print(root.tag)
for name, value in root.items():
    print('%s = %r' % (name, value))

children = root.getchildren()
print ('total number of articles, uncleaned: %i.' % (len(children)) )

# all the titles with their locations
# alltitles = [(i,root[i][0].text) for i in range(1,len(root)) if ":" not in root[i][0].text ]
alltitles = [(i,root[i][0].text) for i in range(1,len(root))]
titles = pd.DataFrame(data = alltitles,columns = ['ind','title' ])

remove = []
for i in range(len(titles)):
# check the NS tag
    node = root[titles.ind[i]]
    if (node[1].text != '0'):  
        # Remove redirect articles
        remove.append(i)
    else:
        if node.find('{http://www.mediawiki.org/xml/export-0.10/}redirect') is not None:
            remove.append(i)
        else:
            for textnode in node[3].iter(tag ='{http://www.mediawiki.org/xml/export-0.10/}text'):
                if len(textnode.text)<ArticleMinLength:
                    remove.append(i)

titles = titles.drop(remove)
# redundant, because loc and iloc differentiate between actual and numerical indices
titles.index = range(len(titles))
print("%d titles dropped \n%d remaining titles" % (len(remove), len(titles) ) )

#column_names = []
#for i in range(0,len(root.getchildren()[1000].getchildren())):
#    column_names.append(root.getchildren()[1000].getchildren()[i].tag)
#colnames = [x[43:] for x in column_names]
#print('colnames ', colnames)
#frame = pd.DataFrame(columns=colnames)

# selecting NumberOfArticles articles randomly
np.random.seed(123)
randomindices = np.random.randint(low=0,high=len(titles),size = NumberOfArticles)
data = titles.iloc[randomindices].copy()
data.index=range(NumberOfArticles)
data.head(3)

# adding text to the data frame
text = [''] * len(data)
for i in data.index:
    for child in root[data.ind[i]]:
        for textnode in child.iter(tag ='{http://www.mediawiki.org/xml/export-0.10/}text'):
            text[i]= textnode.text
            
data.loc[:,'text'] = text
data.head(3) # how does it look?

data.to_pickle('uncleaned-10k-articles.pkl') # for pickle
# data.to_csv('uncleaned-10k-articles.csv') # for csv



