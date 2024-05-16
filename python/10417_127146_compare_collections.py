"""
First we get our script ready with all the libraries we need. If you know Python, you are probably familiar with
most of these files.
"""

### For these import to work, you should have a copy of Anaconda available to you.  We are using Python 3 in this case.
### Also, you will need to install mca and matplotlib_venn.  See the README file for more information.

get_ipython().magic('matplotlib inline')
import os
import csv
import pandas as pd
import numpy as np
from matplotlib_venn import venn2, venn3
import mca
import matplotlib.pyplot as plt
from collections import defaultdict
#import mpld3
#mpld3.enable_notebook()

###################################
## The main class
###################################

class Compare:
    """ 
    Compare -- plot collections for comparison purposes.
    
    Description:
        The purpose of this class is to take a series of collections and plot them so to show how they match.
        If the series is a dictionary, the keys will be used to create plot names.
        If the series contains two or three collections, then the plot will show venn diagrams and return a venn object
        that can be used for other purposes.
        If the series is greater than three, the plot will show the collections in a scatter plot based on correspondence
        scores.
    
    Parameters: 
        @collections (required):  A list of lists or a dict() of size 2 or greater for comparison purposes. 
        @names:  An optional list of names for the collections.  Must be equal in size to collections. If collections is 
            a dict, this parameter will be overwritten.
        @index:  A list of index keys to create a sublist for the collections.
        @var: An optional list for further categorization of the collection data (not fully implemented yet).
        @REMOVE_SINGLES: (Default:True) For 4 collections or more, remove from the analysis any data points that are
            members of only one collection. This reduces the chance that a disproportionately large collection
            will be seen as an outlier merely because it is disproportionately large.
        @DIM3: either 2 or 3 - # of dimensions to visualize (2D or 3D)
        LABEL_BOTH_FACTORS: whether to use two factors in results (default to just one).
            
    """
    
    def __init__ (self, collections, names=[], index=[], var=[], REMOVE_SINGLES=True, DIM3=False, LABEL_BOTH_FACTORS=False):
        self.collection_names = names
        self.index = index
        self.collections = collections
        self.REMOVE_SINGLES = REMOVE_SINGLES
        if DIM3 == True:
            self.DIMS = 3
        else:
            self.DIMS = 2
        self.LABEL_BOTH_FACTORS = LABEL_BOTH_FACTORS
        self.dimensions = None
        self.result = {}
        self.clabels = []
        self.rlabels = []
        self.plabels = []
        #self.cur_depth = self.recur_len(self.collections)
        if isinstance(self.collections, dict):
            # print("dict passed")
            self.collection_names = [x for x in self.collections.keys()]
            self.collections = [x for x in self.collections.values()]
        #print(type([y[0] for y in self.collections][0]))
        # if a dictionary is inputed, then get names from dictionary
        if type([y[0] for y in self.collections][0]) is tuple: #will need to include checks for size of sample 
            print ("yay mca")
            self.collection_names = list(set([x[0] for y in self.collections for x in y]))            
            if self.index:
                self.collections = self.sublist(self.collections, self.index)
                self.collection_names = self.sublist(self.collection_names, self.index)
            self.mca(self.collections, self.collection_names)
        else:            
            #self.collections = dict([(x[0], x[1]) for y in self.collections for x in y])
            if not self.collection_names:
                self.collection_names = range(1, len(self.collections)+1)
            # if index var is provided, use index to filter collection list
            if self.index:
                self.collections = self.sublist(self.collections, self.index)
                self.collection_names = self.sublist(self.collection_names, self.index)
        #two sample venn
            if len(self.collections) == 2:
                self.response = self.two_venn(self.collections)
        #three sample venn
            elif len(self.collections) == 3:
                self.response = self.three_venn(self.collections)
        #use mca for greater than three
            elif len(self.collections) >3:
                if var:
                    self.var = var
                else: 
                    self.var = []
                self.ca = self.ca(self.collections, self.collection_names)
            else:
                self.no_compare()
                
    def recur_len(self, L):
        return sum(L + recur_len(item) if isinstance(item, list) else L for item in L)   
    def no_compare(self):
        return ("Need at least two collections to compare results.")
    #get a sublist from a list of indices
    def sublist (self, list1, list2):
        return([list1[x] for x in list2]) 
    
    ## Data processing functions
    
    def processCollection (self, path):
        #initialise vars:
        urls = []
        #establish the data folder
        for filename in os.listdir(path):
            with open(path+filename, "r") as file:
                print (filename) #see the filenames available.
                urls.append(list({(filename[0:15], line.translate(str.maketrans(')'," ")).split(",")[0][2:6], line.translate(str.maketrans(')'," ")).split(",")[1]) for line in file.readlines()}))
        return(urls)
    
    def convert_full_urls (self, path):
        collection = dict()
        for filename in os.listdir(path):
            with open(path+filename, "r") as file:
                print(filename)
        # result:  {'www.url1.suf', 'www.biglovely.url2.suf', 'education.url3.suf'}
                collect2 = [x.split(".")[-2]+"."+x.split(".")[-1] for x in collect]
                collection[filename[0:15]] = (collect2) #convert collect2 to a dict {truncatedFILENAME: [url1.suf, url2.suf, url3.suf]}
        return (collection)
            
    def convert_subdomain_urls (self, path):
        collection = dict()
        for filename in os.listdir(path):
            with open(path+filename, "r") as file:
                print(filename)
        #split the data by comma and lose the closing url. Put it in a set to remove duplicates.
                collect = {line.translate(str.maketrans(')'," ")).split(",")[1] for line in file.readlines()}
                collect4 = [x for x in collect]
        # result merely converts each set to a big list of urls. (Full scope of analysis)
                collection[filename[0:15]] = (collect4) #convert collect4 to a dict.
        return(collection)

    def removekey(self, d, key):
        r = dict(d)
        del r[key]
        return r
    
    #get set of all items (unduplicated)
    def unionize (self, sets_list):
        return set().union(*sets_list)
    def two_venn (self, collections):
        self.V2_AB = set(collections[0]).intersection(set(collections[1]))
        return  (venn2([set(x) for x in collections], set_labels=self.collection_names))
    def three_venn (self, collections):
        self.V3_ABC = set(collections[0]) & set(collections[1]) & set(collections[2]) 
        self.V3_AB = set(collections[0]) & set(collections[1]) - self.V3_ABC
        self.V3_BC = set(collections[1]) & set(collections[2]) - self.V3_ABC
        self.V3_AC = set(collections[0]) & set(collections[2]) - self.V3_ABC
        self.V3_A = set(collections[0]) - (self.V3_ABC | self.V3_AB | self.V3_AC )
        self.V3_B = set(collections[1]) - (self.V3_ABC | self.V3_AB | self.V3_BC )
        self.V3_C = set(collections[2]) - (self.V3_ABC | self.V3_BC | self.V3_AC )
        return  (venn3([set(x) for x in collections], set_labels=self.collection_names))
    def ca(self, collections, names):
        # use dd to create a list of all websites in the collections
        print (names)
        dd = self.unionize(collections)
        d = [] #create
        e = [] #labels
        fs, cos, cont = 'Factor Score', 'Squared cosines', 'Contributions x 1000'
        #populate table with matches for actors (weblists)
        for y in collections:
            d.append({x: x in y for x in dd})
        #if self.var:
        #    e = ({x.split(".")[0]: x.split(".")[1] for x in dd })
        df = pd.DataFrame(d, index=names)       
        if self.REMOVE_SINGLES:
            df = df.loc[:, df.sum(0) >1 ]
            df.fillna(False)
        #if self.var:
        #    df.loc[:,"SUFFIX"] = pd.Series(e, index=df.index)
        self.response = df.T
        counts = mca.mca(df)
        self.dimensions = counts.L
        print(self.dimensions)
        data = pd.DataFrame(columns=df.index, index=pd.MultiIndex
                      .from_product([[fs, cos, cont], range(1, 3)]))
        self.result["rows"] = counts.fs_r(N=self.DIMS).T
        self.result["columns"] = counts.fs_c(N=self.DIMS).T
            #self.result["df"] = data.T[fs].add(noise).groupby(level=['Collection'])
        #data.loc[fs,    :] = counts.fs_r(N=self.DIMS).T
        points = self.result["rows"]
        urls = self.result["columns"]
        if self.DIMS == 3:
            clabels = data.columns.values
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111, projection='3d')

            plt.margins(0.1)
            plt.axhline(0, color='gray')
            plt.axvline(0, color='gray')
            ax.set_xlabel('Factor 1 (' + str(round(float(self.dimensions[0]), 3)*100) + '%)') 
            ax.set_ylabel('Factor 2 (' + str(round(float(self.dimensions[1]), 3)*100) + '%)')
            ax.set_zlabel('Factor 3 (' + str(round(float(self.dimensions[2]), 3)*100) + '%)')
        
            ax.scatter(*points,  s=120, marker='o', c='r', alpha=.5, linewidths=0)
            ax.scatter(*urls, s=120, marker='s', c='whitesmoke', alpha=.5, linewidths=0)
            for clabel, x, y, z in zip(clabels, *points):
                ax.text(x,y,z,  '%s' % (clabel), size=20, zorder=1, color='k') 
        else:
            self.clabels = data.columns.values
            plt.figure(figsize=(25,25))
            plt.margins(0.1)
            plt.axhline(0, color='gray')
            plt.axvline(0, color='gray')
            plt.xlabel('Factor 1 (' + str(round(float(self.dimensions[0]), 3)*100) + '%)') 
            plt.ylabel('Factor 2 (' + str(round(float(self.dimensions[1]), 3)*100) + '%)')
            plt.scatter(*points,  s=120, marker='o', c='r', alpha=.5, linewidths=0)
            plt.scatter(*urls,  s=120, marker='s', c='whitesmoke', alpha=.5, linewidths=0)
            for label, x, y in zip(self.clabels, *points):
                plt.annotate(label, xy=(x, y), xytext=(x + .03, y + .03))
            if self.LABEL_BOTH_FACTORS:
                self.rlabels = df.T.index
                for label, x, y in zip(self.rlabels, *urls):
                    plt.annotate(label, xy=(x, y), xytext=(x + .03, y + .03))
            plt.show()
        return(data.T)
    
    def mca(self, collections, names):
        #print ([x[2] for y in collections for x in y][0:3])
        default = defaultdict(list)
        coll = defaultdict(list)
        src_index, var_index, d = [], [], []
        for x in collections:
            for y,k,v in x:
                default[y+'%'+k].append(v)
        #print(list(default)[0:3])
        dd = self.unionize([j for y, j in default.items()])
        #print (dd)
        for key, val in default.items():
            #print (key)
            keypair = key.split("%")
            collect, year = keypair[0], keypair[1]
            coll[collect].append(year)
            d.append({url: url in val for url in dd})
        for happy, sad in coll.items():
            src_index = (src_index + [happy] * len(sad))
        #src_index = (happy * len(sad) for happy, sad in coll.items())
            var_index = (var_index + sad)
        col_index = pd.MultiIndex.from_arrays([src_index, var_index], names=["Collection", "Date"])
        #X = {x for x in (self.unionize(collections))}
        table1 = pd.DataFrame(data=d, index=col_index, columns=dd)
        if self.REMOVE_SINGLES:
            table1 = table1.loc[:, table1.sum(0) >1 ]
        table2 = mca.mca(table1)
        #print (table2.index)
        self.response = table1
        self.dimensions = table2.L 
        #print(table2.inertia)
        fs, cos, cont = 'Factor score','Squared cosines', 'Contributions x 1000'
        data = pd.DataFrame(columns=table1.index, index=pd.MultiIndex
                      .from_product([[fs, cos, cont], range(1, self.DIMS+1)]))
        #print(data)
        noise = 0.07 * (np.random.rand(*data.T[fs].shape) - 0.5)
        if self.DIMS > 2:
            data.loc[fs, :] = table2.fs_r(N=self.DIMS).T
            self.result["rows"] = table2.fs_r(N=self.DIMS).T
            self.result["columns"] = table2.fs_c(N=self.DIMS).T
            self.result["df"] = data.T[fs].add(noise).groupby(level=['Collection'])
            
        data.loc[fs,    :] = table2.fs_r(N=self.DIMS).T
 #       print(data.loc[fs, :])

        #print(points)
        urls = table2.fs_c(N=self.DIMS).T
        self.plabels = var_index        

        fs_by_source = data.T[fs].add(noise).groupby(level=['Collection'])

        fs_by_date = data.T[fs]
        self.dpoints = data.loc[fs].values
        print(self.dpoints[1:3])
        fig, ax = plt.subplots(figsize=(10,10))
        plt.margins(0.1)
        plt.axhline(0, color='gray')
        plt.axvline(0, color='gray')
        plt.xlabel('Factor 1 (' + str(round(float(self.dimensions[0]), 3)*100) + '%)')
        plt.ylabel('Factor 2 (' + str(round(float(self.dimensions[1]), 3)*100) + '%)')
        ax.margins(0.1)
        markers = '^', 's', 'o', 'o', 'v', "<", ">", "p", "8", "h"
        colors = 'r', 'g', 'b', 'y', 'orange', 'peachpuff', 'm', 'c', 'k', 'navy'
        for fscore, marker, color in zip(fs_by_source, markers, colors):
            #print(type(fscore))
            label, points = fscore
            ax.plot(*points.T.values[0:1], marker=marker, color=color, label=label, linestyle='', alpha=.5, mew=0, ms=12)
            for plabel, x, y in zip(self.plabels, *self.dpoints[1:3]):
                print(plabel)
                #print(xy)
                plt.annotate(plabel, xy=(x, y), xytext=(x + .15, y + .15))
        ax.legend(numpoints=1, loc=4)
        plt.show()
        

#initialise vars:
collection = dict()
collection_2 = dict()
var = dict()

#establish the data folder
path = "assembled/"

#get the files

print ("These are the files that have been accessed through this script:\n\n")
for filename in os.listdir(path):
    with open(path+filename, "r") as file:
        print(filename)
        #split the data by comma and lose the closing url. Put it in a set to remove duplicates.
        collect = {line.translate(str.maketrans(')'," ")).split(",")[1] for line in file.readlines()}
        # result:  {'www.url1.suf', 'www.biglovely.url2.suf', 'education.url3.suf'}
        collect2 = [x.split(".")[-2]+"."+x.split(".")[-1] for x in collect]
        # result: ['url1.suf', 'url2.suf', 'url3.suf']  (this decreases scope of analysis - removes "education" 
        # in education.ab.ca), for example.
        collect4 = [x for x in collect]
        # result merely converts each set to a big list of urls. (Full scope of analysis)
        collection[filename[0:10]] = (collect2) #convert collect2 to a dict {truncatedFILENAME: [url1.suf, url2.suf, url3.suf]}
        collection_2[filename[0:10]] = (collect4) #convert collect4 to a dict.

#Just separate the names and values for now.
comparit = [x for x in collection_2.values()]
names = [x.upper() for x in collection_2.keys()]

""" 
Since the variable "collection" has 16 different archives, we can use the index variable to choose two. When you 
have two collections, then Compare will provide you with a venn diagram with two variables.

You can find out the content inside the circles by using V2_[A, B or AB].


(V2_A (V2_AB) V2_B)

"""

#Two collections will produce a two-way Venn diagram showing the cross over in terms of links.
#Since collection is a dict() no need to include names.
compare1 = Compare(collection, index=[4,7])
print("Links in Common (there are "+ str(len(compare1.V2_AB)) + ") : /n" )
for x in compare1.V2_AB:
    print (str(x) + "\n")

#Although you can add your own names if you want to ...  (recall "names" is x.upper()))

compare1 = Compare(collection, names=names, index=[4,7] )
print("Links that both collections have in common: " + ", ".join(compare1.V2_AB))

# What happens with three collections
compare2 = Compare(comparit, names, [2,0,5])
print("Links that all collections have in common: " + ', '.join(compare2.V3_ABC))

# With more than three collections, the output switches to correspondence analysis.
# Katherine Faust offers a great overview of the method here:
# www.socsci.uci.edu/~kfaust/faust/research/articles/articles.htm

# In this case, we've eliminated a few items from the analysis.

compare3 = Compare(collection, names, [i for i, x in enumerate(collection) if i not in [5,6,7,14,16,17,18,19,21,23,24]])

"""
Inertia is somewhat like an R-squared score for a correspondence graph.  The overall inertia is only 40%, so quite low.
However, the third and fourth dimensions of analysis also seem about as relevant and the first and second.

Later developments will show how to look at the correspondence from these perspectives as well.
"""

# Compare.dimensions store factor values for all the dimensions.  The top two dimensions are shown in the graph above.
# Inertia or the total explantory value of the graph is the sum of all these values.
print ([round(x,3) for x in compare3.dimensions])
print ("Inertia: " + str(sum(compare3.dimensions)))

# Compare.response stores the table that these scores are based on (I've selected only 5 items for clarity)
#If REMOVE_SINGLES is not FALSE, then no row should contain fewer than two "TRUES".

print (compare3.response[10:15])

"""
Let's do the same thing with a different dataset:
 
 Name Entity Recognition (NER) offers a way of identifying locations named within the collections.  
 So we have data in csv format like this:
 
CPP 201508,Sum of Frequency
Canada ,283162
Ontario ,34197
United States,32008
Ottawa ,30233
Toronto ,18787
Alberta,17015
British Columbia ,13181
Manitoba,11332
Hawaiian ,10110
QUEBEC,9633
Vancouver,9442

"""


#initialise vars:
loc_collection = dict()
loc_collection_2 = dict()
loc_var = dict()

#establish the data folder
loc_path = "../../NER/"

#get the files
for loc_filename in os.listdir(loc_path):
    with open(loc_path+loc_filename, "r", encoding="utf-8", errors="ignore") as loc_file:
        print(loc_filename)
        loc_collect = [row[0] for row in csv.reader(loc_file)]
        loc_collection[loc_filename[0:10]] = (loc_collect)


#Just separate the names and values for now.
loc_comparit = [x for x in loc_collection.values()]
loc_names = [x.upper() for x in loc_collection.keys()]

"""  This provides a graph that shows the dates (labels) in action by collection (shape/color).  It is also possible to show the 
collections by date if desired.
"""

compare4 = Compare(urls, index=[0,1,2,3], DIMENSIONS_OP=3)
#print(compare4.dimensions)
#print(compare4.response[0:10])

loc_compare_1 = Compare(loc_collection)
#print ("Locations in all Three collections: \n\n" + ", ".join(loc_compare_1.V3_ABC) + "\n ... \n")
#print ("in Just the First (top Left): \n\n" + ", ".join(loc_compare_1.V3_A) + "\n ... \n")
#print ("in top left and bottom \(purple section\): \n\n" + ", ".join(loc_compare_1.V3_AC) + "\n ... \n")
#print ("in top right and bottom \(light blue\): \n\n" + ", ".join(loc_compare_1.V3_BC) + "\n ... \n")

""" 
Now let\'s try to use the dates in our analysis.  This brings up the power a little to use multiple correspondence 
analysis.

((201601,linkis.com),11102), 
((201601,m.youtube.com),8764),
((201601,www.youtube.com),7481)

"""

#initialise vars:
dat_collection = dict()
dat_collection_2 = dict()
dat_var = dict()
dat_collect = dict()
#establish the data folder
dat_path = "assembled/"
urls = []

#get the files
for dat_filename in os.listdir(dat_path):
    with open(dat_path+dat_filename, "r") as dat_file:
        print (dat_filename)
        urls.append(list({(dat_filename[0:10], line.translate(str.maketrans(')'," ")).split(",")[0][2:6], line.translate(str.maketrans(')'," ")).split(",")[1]) for line in dat_file.readlines()}))

""" This produces something like this:
[ { (COLLECTION_NAME, DATE, URL)}]  (the {} is to make it unique.)

"""

#Unit tests to be removed later.

import unittest

class CompareTests(unittest.TestCase):
    
    collection1 = ["google", "apple", "microsoft", "msn", "napster", "oracle", "amazon", "ibm"]
    collection2 = ["google", "pear", "thebeatles", "thepogues", "napster", "apple", "cow"]
    collection3 = ["google", "apple", "msn", "skunk", "beaver", "wolf", "cow"]
    collection4 = ["apple", "jump", "walk", "run", "saunter", "skunk", "napster"]
    collection5 = ["pear", "wolf", "jive", "tango"]
    collection6 = ["google", "apple", "msn", "thepogues", "napster", "wolf", "amazon", "tango"]
    one_collect = [collection1]
    two_collect = [collection1, collection2]
    three_collect = [collection1, collection2, collection3]
    all_collect = [collection1, collection2, collection3, collection4, collection5, collection6]
    
    def test_one (self):
        print("test error with one collection")
        self.assertTrue(Compare(self.one_collect), "Need at least two collections to compare results.")
        
    def test_two (self):
        print ("test results for two collections")
        self.assertTrue(Compare(self.two_collect).response.subset_labels[1].get_text(), 4)
        self.assertTrue(Compare(self.two_collect).response.subset_labels[0].get_text(), 5)
        self.assertTrue(Compare(self.two_collect).response.subset_labels[2].get_text(), 3)
    
    def test_three (self):
        print("test results for three")
        self.assertTrue(Compare(self.three_collect).response.subset_labels[0].get_text(), 4)
        self.assertTrue(Compare(self.three_collect).response.subset_labels[1].get_text(), 3)
        self.assertTrue(Compare(self.three_collect).response.subset_labels[0].get_text(), 1)
        self.assertTrue(Compare(self.three_collect).response.subset_labels[0].get_text(), 3)
        self.assertTrue(Compare(self.three_collect).response.subset_labels[0].get_text(), 1)
        self.assertTrue(Compare(self.three_collect).response.subset_labels[0].get_text(), 1)
        self.assertTrue(Compare(self.three_collect).response.subset_labels[0].get_text(), 2)
    
    def test_all (self):
        print("test results for more than three")
        test=Compare(self.all_collect, names=["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"], REMOVE_SINGLES=False)
        self.assertTrue(list(Compare(self.all_collect, names=["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"], REMOVE_SINGLES=False).response.iloc[1].values), 
                        [True, True, True, True, False, True])
        self.assertTrue(list(Compare(self.all_collect, names=["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"], REMOVE_SINGLES=False).response.ix['amazon'].values),
                        [True, False, False, False, False, True])
        self.assertTrue(list(Compare(self.all_collect, names=["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX"], REMOVE_SINGLES=False).response.iloc[5].values),
                        [True, False, False, False, False, False])
        
        
        

suite = unittest.TestLoader().loadTestsFromTestCase(CompareTests)
unittest.TextTestRunner().run(suite)
        


compare3.ca.loc[:,'SUFFIX'] = pd.Series(var, index=compare3.ca.index)
print (compare3.ca)
compare3.ca.to_csv("output.csv",  encoding='utf-8')



