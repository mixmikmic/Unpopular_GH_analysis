#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

#constants
get_ipython().magic('matplotlib inline')
sns.set_style("dark")

mainFrame = pd.read_csv("../data/processed/procDataset.csv")

#plot out gender
mainFrame["responseID"] = range(mainFrame.shape[0])
genderCountFrame = mainFrame.groupby("gender",
                                     as_index = False)["responseID"].count()
genderCountFrame = genderCountFrame.rename(columns = {"responseID":"count"})
sns.barplot(x = "gender",y = "count",data = genderCountFrame)
plt.xlabel("Personally Identified Gender")
plt.ylabel("Count")
plt.title("Distribution of\nPersonally Identified Gender")
plt.savefig("../reports/blogPost/blogPostFigures/figure5.png",
            bbox_inches = "tight")

crossTab = pd.crosstab(mainFrame["gender"],mainFrame["diagnosedWithMHD"],
                       normalize = "index")
crossTab.columns.name = "Diagnosed?"
display(crossTab)



