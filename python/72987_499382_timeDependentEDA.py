#imports

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

#helpers

get_ipython().magic('matplotlib inline')
sns.set_style("dark")
sigLev = 3

#import dataset
timeDepFrame = pd.read_csv("../../data/processed/procTimeDataset.csv")

timeCountFrame = timeDepFrame.groupby("year",as_index = False)["gender"].count()
timeCountFrame = timeCountFrame.rename(columns = {"gender":"count"})
display(HTML(timeCountFrame.to_html(index = False)))

diagTimeCountFrame = timeDepFrame.groupby(["diagnosedWithMHD","year"],
                                as_index = False)["gender"].count()
diagTimeCountFrame = diagTimeCountFrame.rename(columns = {"gender":"count"})
#get density
sns.barplot(x = "diagnosedWithMHD",y = "count",hue = "year",
            data = diagTimeCountFrame)
plt.xlabel("Diagnosed with Mental Health Condition")
plt.ylabel("Count")
plt.title("Distribution of Diagnosed With Mental Health Condition\nOver Time")
plt.savefig("../../reports/thirdBlogPost/figures/figure02.png")

sns.boxplot(x = "year",y = "age",data = timeDepFrame)
plt.xlabel("Year")
plt.ylabel("Age")
plt.title("Age on Year")
plt.savefig("../../reports/thirdBlogPost/figures/figure03.png")

genderTimeCountFrame = timeDepFrame.groupby(["gender","year"],as_index = False)[
                                                "age"].count()
genderTimeCountFrame = genderTimeCountFrame.rename(columns = {"age":"count"})
sns.barplot(x = "gender",y = "count",hue = "year",data = genderTimeCountFrame)
plt.xlabel("Gender")
plt.ylabel("Count")
plt.title("Distribution of Gender by Year")
plt.savefig("../../reports/thirdBlogPost/figures/figure04.png")

sizeTimeFrame = timeDepFrame.groupby(["companySize","year"],as_index = False)[
                                "gender"].count()
sizeTimeFrame = sizeTimeFrame.rename(columns = {"gender":"count"})
print sizeTimeFrame

#reorganize into desired shape
sizeTimeFrame = sizeTimeFrame.iloc[[0,1,8,9,4,5,2,3,6,7,10,11],:]
#then plot
sns.barplot(x = "companySize",y = "count",hue = "year",data = sizeTimeFrame)

sns.boxplot(x = "diagnosedWithMHD",y = "age",hue = "year",data = timeDepFrame)

timeDepFrame["diagnosedWithMHD"] = timeDepFrame["diagnosedWithMHD"].map(
                                                {"Yes":1,"No":0})

pd.crosstab([timeDepFrame["year"],timeDepFrame["gender"]],
            timeDepFrame["diagnosedWithMHD"],normalize = "index")

#generate isUSA
timeDepFrame["isUSA"] = 0
timeDepFrame.loc[(timeDepFrame["workCountry"] == "United States of America") |
                 (timeDepFrame["workCountry"] == "United States"),
                 "isUSA"] = 1

pd.crosstab([timeDepFrame["year"],timeDepFrame["isUSA"]],
            timeDepFrame["diagnosedWithMHD"],normalize = "index")

timeDepFrame["isUK"] = 0
timeDepFrame.loc[timeDepFrame["workCountry"] == "United Kingdom",
                 "isUK"] = 1

pd.crosstab([timeDepFrame["year"],timeDepFrame["isUK"]],
            timeDepFrame["diagnosedWithMHD"],normalize = "index")

timeDepFrame["isCA"] = 0
timeDepFrame.loc[timeDepFrame["workCountry"] == "Canada",
                 "isCA"] = 1

pd.crosstab([timeDepFrame["year"],timeDepFrame["isCA"]],
            timeDepFrame["diagnosedWithMHD"],normalize = "index")

timeDepFrame.to_csv("../../data/processed/timeDataset_withCountryDummies.csv",
                    index = False)



