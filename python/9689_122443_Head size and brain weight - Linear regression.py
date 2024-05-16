get_ipython().magic('pylab inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("http://www.stat.ufl.edu/~winner/data/brainhead.dat", sep=" ", skipinitialspace=True, header=None)

df.head()

df.rename(columns={0:"gender",1:"age group",2:"head size", 3:"brain weight"}, inplace=True)

df.head()

youngmen = df[df["gender"]==1][df["age group"]==1]

plt.scatter(youngmen["head size"], youngmen["brain weight"])
plt.xlabel("head size (cm^3)")
plt.ylabel("brain weight (g)")
plt.title("Brain weight vs. head size")

oldmen = df[df["gender"]==1][df["age group"]==2]

plt.scatter(youngmen["head size"], youngmen["brain weight"])
plt.scatter(oldmen["head size"], oldmen["brain weight"],color="r")
plt.xlabel("head size (cm^3)")
plt.ylabel("brain weight (g)")
plt.title("Brain weight vs. head size for men")
plt.legend(["age 20-46","age 46+"],loc="upper left")

youngwomen = df[df["gender"]==2][df["age group"]==1]
oldwomen = df[df["gender"]==2][df["age group"]==2]

plt.scatter(youngwomen["head size"], youngwomen["brain weight"])
plt.scatter(oldwomen["head size"], oldwomen["brain weight"],color="r")
plt.xlabel("head size (cm^3)")
plt.ylabel("brain weight (g)")
plt.title("Brain weight vs. head size for women")
plt.legend(["age 20-46","age 46+"],loc="upper left")

plt.scatter(youngmen["head size"], youngmen["brain weight"])
plt.scatter(oldmen["head size"], oldmen["brain weight"],color="r")
plt.scatter(youngwomen["head size"], youngwomen["brain weight"], color="g")
plt.scatter(oldwomen["head size"], oldwomen["brain weight"],color="m")
plt.xlabel("head size (cm^3)")
plt.ylabel("brain weight (g)")
plt.title("Brain weight vs. head size for women")
plt.legend(["men 0-46","men 46+", "women 20-46","women 46+"],loc="upper left")

allmen = df[df["gender"]==1]

allmen["head size"].describe()

allmen["brain weight"].describe()

plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
plt.hist(allmen["head size"],bins=20,normed=True);
plt.xlabel("Head size (cm^3)")
plt.title("Distribution of male head sizes");
plt.subplot(1,2,2)
plt.hist(allmen["brain weight"],bins=20,normed=True);
plt.xlabel("Brain weight (g)")
plt.title("Distribution of male brain weights");

allwomen = df[df["gender"]==2]

allwomen["head size"].describe()

allwomen["brain weight"].describe()

plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
plt.hist(allwomen["head size"],bins=20,normed=True);
plt.xlabel("Head size (cm^3)")
plt.title("Distribution of female head sizes");
plt.subplot(1,2,2)
plt.hist(allwomen["brain weight"],bins=20,normed=True);
plt.xlabel("Brain weight (g)")
plt.title("Distribution of female brain weights");

from sklearn import linear_model

regr_men = linear_model.LinearRegression()

regr_men.fit(allmen["head size"].reshape(-1,1), allmen["brain weight"])

plt.scatter(allmen["head size"], allmen["brain weight"])
plt.plot(allmen["head size"], regr_men.predict(allmen["head size"].reshape(-1,1)), color='red',linewidth=3)
plt.title("Brain weight vs. head size for all men")
plt.legend(["data","linear fit"],loc="upper left")

regr_women = linear_model.LinearRegression()

regr_women.fit(allwomen["head size"].reshape(-1,1), allwomen["brain weight"])

plt.scatter(allwomen["head size"], allwomen["brain weight"])
plt.plot(allwomen["head size"], regr_women.predict(allwomen["head size"].reshape(-1,1)), color='red',linewidth=3)
plt.title("Brain weight vs. head size for all women")
plt.legend(["data","linear fit"],loc="upper left")



