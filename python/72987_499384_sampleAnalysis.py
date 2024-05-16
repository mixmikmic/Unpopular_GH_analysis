#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#helpers
sigLev = 3
get_ipython().magic('matplotlib inline')
sns.set_style("whitegrid")
pd.options.display.precision = sigLev

#load in data
sampleFrame = pd.read_csv("../data/raw/sampleOfComments.csv")

sampleFrame.shape

sampleFrame.columns

#data quality check
numNullFrame = sampleFrame.apply(lambda x: x[x.isnull()].shape[0],axis = 0)
numNullFrame

#get number of levels
nUniqueFrame = sampleFrame.apply(lambda x: x.nunique(),axis = 0)
nUniqueFrame

bodyFrame = sampleFrame.groupby("body",as_index = False)["score"].count()
bodyFrame = bodyFrame.rename(columns = {"score":"count"})
bodyFrame = bodyFrame.sort_values("count",ascending = False)
bodyFrame

filteredSampleFrame = sampleFrame[~(sampleFrame["body"].isin([
                                    "[removed]","[deleted]"]))]
filteredSampleFrame.shape

parentFrame = filteredSampleFrame.groupby("parent_id",as_index = False)[
                                                            "score"].count()
parentFrame = parentFrame.rename(columns = {"score":"count"})
parentFrame = parentFrame.sort_values("count",ascending = False)
parentFrame

plt.hist(parentFrame["count"],)
plt.xlabel("Number at Parent")
plt.ylabel("Count")
plt.title("Distribution of\nParent ID Frequencies")



