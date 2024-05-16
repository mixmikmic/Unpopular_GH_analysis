get_ipython().magic('matplotlib inline')
from Compare import Compare # The Compare class
import os #file operations
from collections import defaultdict #complex dictionaries will eventually be moved to the Compare class.
import matplotlib.pyplot as plt #plotting library
import copy #need deepcopy() for working with the dictionaries.
import pandas as pd
from adjustText import adjustText as AT

path = "ALL/"

##  INCLUDE ANY DICTIONARIES YOU WISH TO EXCLUDE in the following list.  The excluded libraries will be removed from
##  newdict2 for comparison purposes:

exclude1 = ['DAL_Mikmaq', 'WINNIPEG_truth_', 'SFU_BC_Local_go', 'SFU_NGOS', 'SFU_PKP',  'DAL_NS_Municipl', 'DAL_HRM_Docs', 
            'WINNIPEG_oral_h',  'WINNIPEG_websit', 'WINNIPEG_digiti'] 

dummies = ['DUMMY_OVERALL', 'DUMMY_MEDIA', 'DUMMY_GOVERNMEN', 'DUMMY_ORGANIZAT',
           'DUMMY_TECHNOLOG', 'DUMMY_SOCIALMED']

def processCollection (path):
    #initialise vars:
    urls = []
    #establish the data folder
    for filename in os.listdir(path):
        with open(path+filename, "r") as file:
            print (filename) #see the filenames available.
            urls.append(list({(filename[0:15], line.translate(str.maketrans(')'," ")).split(",")[0][2:6], line.translate(str.maketrans(')'," ")).split(",")[1].strip()) for line in file.readlines()}))
    return(urls)

newdict = defaultdict(lambda: defaultdict(list))

PC = processCollection(path)
for collect in PC:
    for coll, date, url in collect:
        if coll in dummies:
            pass
        else:
            newdict[date][coll].append(url)
            
## add_two_collections merges two dictionaries and is used by reduce_collections to show the accumulation of 
## the collections over dates.

def add_two_collections (col1, col2):
    # This takes two collections and combines them into one.
    col_1 = col1.copy()
    for coll, values in col2.items():
        #print(values)
        try:
            col_1[coll] = set(col_1[coll])
            col_1[coll].update(set(values)) 
            col_1[coll] = list(col_1[coll])
        except KeyError:
            col_1[coll] = list(values)       
    return col_1

## reduce_collections takes the newdict dictionaries in {date: collection : [list of urls]} form
## and returns a list of the dictionaries as they accumulated by date. [2009 : collection [list of urls], 
## 2010+2009 : collection etc.]
def reduce_collections (dictionary):
    dict_list = []
    fulllist = {}
    dict2 = copy.deepcopy(dictionary)
    for x, y in sorted(dict2.items()):
        #print(x)
        n = dictionary.pop(x)
        if len(dict_list) < 1:
            dict_list.append(n)
        #print(n)
        else:
            dict_list.append((add_two_collections(dict_list[-1], n)))
        #print(dict_list)
    return(dict_list)

collection_var = reduce_collections(copy.deepcopy(newdict))

# Collection var is a list of dictionaries starting from the earliest to the latest. The later dictionaries
# are accumulations of the former.

dummy = Compare(collection_var[-1])

# Create a new defaultdict and a list of collections to exclude

exclude = ['ALBERTA_heritag', 'ALBERTA_hcf_onl']
newdict2 = defaultdict(lambda: defaultdict(list))

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll in exclude or coll in dummies:
            pass
        else:
            newdict2[date][coll].append(url)

collection_var2 = reduce_collections (copy.deepcopy(newdict2))
dummy2 = Compare(collection_var2[-1])

d2points = dummy2.result['rows']
d2urls = dummy2.result['columns']
d2rlabels = dummy2.response.index
d2labels = dummy2.collection_names
plt.figure(figsize=(10,10))
plt.margins(0.1)
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.scatter(*d2points,  s=120, marker='o', c='r', alpha=.5, linewidths=0)
plt.scatter(*d2urls,  s=120, marker='s', c='whitesmoke', alpha=.5, linewidths=0)
texts=[]
for x, y, label in zip(*d2points, d2labels):
    texts.append(plt.text(x, y, label))

AT.adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

plt.show()

Cluster = Compare([collection_var2[-1]['ALBERTA_floods_'], collection_var2[-1]['ALBERTA_energy_'], 
                  collection_var2[-1]['ALBERTA_univers']], names=["floods", "energy", "university"])

print("These sites were common to all three collections." + str(Cluster.V3_ABC) + "\n")
print("These sites were common to floods and energy only" + str(Cluster.V3_AB) + "\n")
print("These sites were common to energy and university" + str(Cluster.V3_AC) + "\n")
print("These sites were common to energy and floods" + str(Cluster.V3_BC) + "\n")

Cluster2 = Compare([collection_var[-1]['ALBERTA_heritag'], collection_var[-1]['ALBERTA_hcf_onl'], collection_var[-1]['ALBERTA_prairie']], names=["HERITAGE", "HCF_ENCYCL", "PRAIRIES"])

print("These sites were common to all three collections." + str(Cluster2.V3_ABC) + "\n")
print("These sites were common to heritage and hfc encyclopedia only" + str(Cluster2.V3_AB) + "\n")
print("These sites were common to heritage and prairies" + str(Cluster2.V3_AC) + "\n")
print("These sites were common to hcf encyclopedia and prairies" + str(Cluster2.V3_BC) + "\n")

exclude = ['ALBERTA_heritag', 'ALBERTA_hch_onl']
newdict2 = defaultdict(lambda: defaultdict(list))

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll in exclude or coll in dummies[1:6]:
            pass
        else:
            newdict2[date][coll].append(url)

collection_var3 = reduce_collections (copy.deepcopy(newdict2))
dummy3 = Compare(collection_var3[-1])

exclude = ['ALBERTA_heritag', 'ALBERTA_hch_onl']
newdict2 = defaultdict(lambda: defaultdict(list))

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll in exclude or coll in dummies[0]:
            pass
        else:
            newdict2[date][coll].append(url)

collection_var3 = reduce_collections (copy.deepcopy(newdict2))
dummy3 = Compare(collection_var3[-1])

d3points = dummy3.result['rows']
d3urls = dummy3.result['columns']
d3rlabels = dummy3.response.index
d3labels = dummy3.collection_names
plt.figure(figsize=(10,10))
plt.margins(0.1)
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.scatter(*d3points,  s=120, marker='o', c='r', alpha=.5, linewidths=0)
plt.scatter(*d3urls,  s=120, marker='s', c='whitesmoke', alpha=.5, linewidths=0)
texts=[]
for x, y, label in zip(*d3points, d3labels):
    texts.append(plt.text(x, y, label))

AT.adjust_text(texts, arrowprops=dict(arrowstyle="-", color='r', lw=0.5))

plt.show()

exclude = ['ALBERTA_heritag', 'ALBERTA_hch_onl']
newdict2 = defaultdict(lambda: defaultdict(list))

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll in exclude or coll in [x for i,x in enumerate(dummies) if i!=2]: #Government is the 3rd in the dummies list.
            pass
        else:
            newdict2[date][coll].append(url)

collection_var3 = reduce_collections (copy.deepcopy(newdict2))
dummy3 = Compare(collection_var3[-1])

exclude = ['ALBERTA_heritag', 'ALBERTA_hch_onl']
newdict2 = defaultdict(lambda: defaultdict(list))

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll in exclude or coll in [x for i,x in enumerate(dummies) if i!=1]: #Government is the 3rd in the dummies list.
            pass
        else:
            newdict2[date][coll].append(url)

collection_var3 = reduce_collections (copy.deepcopy(newdict2))
dummy3 = Compare(collection_var3[-1])

newpath = "frequencies/"

textdict = dict()

def processText (path):
    #initialise vars:
    text = []
    names = []
    lines= []
    #establish the data folder
    for filename in os.listdir(path):
        with open(path+filename, "r") as file:
            print (filename) #see the filenames available.
            lines = [line.strip().split(" ")[1] if len(line.strip().split(" ")[1]) > 4 else "none" for line in file.readlines()[0:250] if len(line.strip().split(" ")) == 2]
        text.append(lines)
        names.append(filename[0:25])
                
            #text.append(list({(filename[0:25], line.translate(str.maketrans(')'," ")).split(",")[0][2:6], line.translate(str.maketrans(')'," ")).split(",")[1].strip()) for line in file.readlines()[0:5]}))
    return([text, names])

TC = processText(newpath)
#print(TC)

CP = Compare(TC[0], names=TC[1], LABEL_BOTH_FACTORS=True)


points = CP.result['rows']
urls = CP.result['columns']
rlabels = CP.response.index
labels = CP.collection_names
plt.figure(figsize=(10,10))
plt.margins(0.1)
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.scatter(*points,  s=120, marker='o', c='r', alpha=.5, linewidths=0)
plt.scatter(*urls,  s=120, marker='s', c='b', alpha=.5, linewidths=0)
texts=[]
rtexts=[]
for x, y, label in zip(*points, labels):
    texts.append(plt.text(x, y, label, color='b'))

for rx, ry, rlabel in zip(*urls, rlabels):
    rtexts.append(plt.text(rx, ry, rlabel))
AT.adjust_text(rtexts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

plt.show()

newpath = "frequencies/"

textdict = dict()

def processText (path):
    #initialise vars:
    text = []
    names = []
    lines= []
    #establish the data folder
    for filename in os.listdir(path):
        with open(path+filename, "r") as file:
            print (filename) #see the filenames available.
            lines = [line.strip().split(" ")[1] if len(line.strip().split(" ")[1]) > 4 else "none" for line in file.readlines()[0:250] if len(line.strip().split(" ")) == 2]
        text.append(lines)
        names.append(filename[0:25])
                
            #text.append(list({(filename[0:25], line.translate(str.maketrans(')'," ")).split(",")[0][2:6], line.translate(str.maketrans(')'," ")).split(",")[1].strip()) for line in file.readlines()[0:5]}))
    return([text, names])

TC = processText(newpath)
#print(TC)

CP = Compare(TC[0], names=TC[1], LABEL_BOTH_FACTORS=True)

points = CP.result['rows']
urls = CP.result['columns']
rlabels = CP.response.index
labels = CP.collection_names
plt.figure(figsize=(10,10))
plt.margins(0.1)
plt.axhline(0, color='gray')
plt.axvline(0, color='gray')
plt.scatter(*points,  s=120, marker='o', c='r', alpha=.5, linewidths=0)
plt.scatter(*urls,  s=120, marker='s', c='b', alpha=.5, linewidths=0)
texts=[]
rtexts=[]
for x, y, label in zip(*points, labels):
    texts.append(plt.text(x, y, label, color='b'))

for rx, ry, rlabel in zip(*urls, rlabels):
    rtexts.append(plt.text(rx, ry, rlabel))
AT.adjust_text(rtexts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))

plt.show()

# Create a new defaultdict and a list of collections to exclude

exclude = ['ALBERTA_heritag', 'ALBERTA_hcf_onl']
newdict2 = defaultdict(lambda: defaultdict(list))

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll in exclude or coll in exclude1 or coll in dummies[0]:
            pass
        else:
            newdict2[date][coll].append(url)

collection_var2 = reduce_collections (copy.deepcopy(newdict2))
dummy2 = Compare(collection_var2[-1])



