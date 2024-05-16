#First let's import the necessary libraries.
get_ipython().magic('matplotlib inline')
from Compare import Compare # The Compare class
import os #file operations
from collections import defaultdict #complex dictionaries
import matplotlib.pyplot as plt #plotting library
from mpl_toolkits.mplot3d import Axes3D #for 3d graphs
import copy #need deepcopy() for working with the dictionaries.

####   Uncomment the items below if you want 
####   to use D3 for output.

#import mpld3
#mpld3.enable_notebook()

# The output below takes the derivative files from the folder "UFT/" and puts them into a python dictionary (array)
# for later use.  I have included two of these. One including the Heritage Community Foundation's collection and one
# not.

path = "SFU/"

def processCollection (path):
    #initialise vars:
    urls = []
    #establish the data folder
    for filename in os.listdir(path):
        with open(path+filename, "r") as file:
            print (filename) #see the filenames available.
            urls.append(list({(filename[0:15], line.translate(str.maketrans(')'," ")).split(",")[0][2:6], line.translate(str.maketrans(')'," ")).split(",")[1].strip()) for line in file.readlines()}))
    return(urls)

#newdict = defaultdict(dict)
newdict = defaultdict(lambda: defaultdict(list))
newdict2 = defaultdict(lambda: defaultdict(list))
PC = processCollection(path)
#print(list(zip(PC[0])))
#print(list(zip(PC[0][0])))
#print (**collect)
for collect in PC:
    for coll, date, url in collect:
        newdict[date][coll].append(url)

# newdict will provide all the data like so:

#newdict2 eliminates collections in if-then statement.
for collect in PC:
    for coll, date, url in collect:
        if coll == 'ALBERTA_heritag' or coll=='DUMMY_GOVERNMEN' or coll == 'ALBERTA_hcf_onl' or coll=="DUMMY_MEDIA" or coll=="DUMMY" or coll== "ORGDUMMY" or coll== "TECHDUMMY" or coll== "SOCIALMEDIADUMM" or coll=="GOVDUMMY":
            pass
        else:
            newdict2[date][coll].append(url)

#{'DATE': {'COLLECTION': ['url1.com', 'url2.com', 'etc']}}
#

 

## Produce a dictionary output that creates a list of outputs suitable for analysis by date.
##
## collection_var[-1] would analyze all the links together until the latest year (2016). collection_var[-2]
## would analyze everything up to t-1 (2015).
##
## Our hope for the future is that the data could be used in an animation, showing changes over time. But for now, 
## we will just show the progress.

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

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

collection_var = reduce_collections (copy.deepcopy(newdict2))

# Collection var is a list of dictionaries starting from the earliest to the latest. The later dictionaries
# are accumulations of the former.

x = Compare({"happy": ["ecstatic", "bursting", "nostalgic"], "sad": ["down", "depressed", "nostalgic"]}, LABEL_BOTH_FACTORS=True)

SFU = Compare(collection_var[-1])

#collection_var[-3] = removekey(collection_var[-3], 'SOCIALMEDIA')
collection_var[-1]['MEDIADUMMY'] = newdict['2015']['DUMMY_MEDIA']

SFU = Compare(collection_var[-1])

SFU = Compare(collection_var[-1])

SFU = Compare(collection_var[-1])

#collection_var[-1] = removekey(collection_var[-1], 'WAHR_ymmfire-ur')
#collection_var[-1] = removekey(collection_var[-1], 'WAHR_panamapape')
#collection_var[-1] = removekey(collection_var[-1], 'WAHR_exln42-all')
#collection_var[-1] = removekey(collection_var[-1], 'MEDIADUMMY')
Compare(collection_var[-1])

TestDict1 = {'2009': {'c1': {'lk1', 'lk2', 'lk3'},
                     'c2': {'lk1', 'lk10', 'lk20', 'lk2'},
                     'c3': {'lk3', 'lk10', 'lk33', 'lk4'}},
            '2010': {'c1': {'lk3', 'lk5', 'lk6'},
                    'c3': {'lk10', 'lk9', 'lk7'}},
            '2011': {'c1': {'lk3', 'lk5', 'lk6'},
                    'c4': {'lk1', 'lk2', 'lk3'}},
            '2012': {'c1': {'lk1', 'lk99', 'lk6'}}
           }

#print(list(zip(*zip(TestDict['2009'])))

