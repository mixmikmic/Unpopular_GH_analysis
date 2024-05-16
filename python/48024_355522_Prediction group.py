# json
import json

# math
import math

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

# Random libraries and seeds:
import random
random.seed(2)
np.random.seed(2)

pd.set_option('display.max_columns', None)

# From: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

import sklearn.model_selection as mds
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

def get_accuracy(number_groups, region):
    
    #print("datasets/%d/%s.csv" % (number_groups, region))
    data = pd.read_csv("datasets/%d/%s.csv" % (number_groups, region))
    
    data = pd.get_dummies(data, columns = ["attacktype1_txt",
                     "targtype1_txt",
                     "weaptype1_txt",
                     "natlty1_txt",
                     "weaptype1_txt",
                     "weapsubtype1_txt"])
    
    train, validate, test = np.split(data.sample(frac=1, random_state = 2), [int(.6*len(data)), int(.8*len(data))])
    
    X_train = train.drop(["gname", "region_txt"], axis=1)
    Y_train = train["gname"]
    
    X_val = validate.drop(["gname", "region_txt"], axis=1)
    Y_val = validate["gname"]
    
    X_test = test.drop(["gname", "region_txt"], axis=1)
    Y_test = test["gname"]
    
    # 70% train, 30% test
    #msk = np.random.rand(len(data_region)) < 0.7
    
    #X_train = X[msk]
    #Y_train = Y[msk]
    
    #X_test = X[~msk]
    #Y_test = Y[~msk]
    
    model = OneVsRestClassifier(RandomForestClassifier(random_state=2)).fit(X_train, Y_train)
    
    Y_pred = model.predict(X_val)
    
    return(model, (sum(Y_pred == Y_val) / len(Y_pred)), X_train, Y_train, X_val, Y_val, X_test, Y_test) # return accuracy

    #print("%s, %d/%d => %s" % (region, sum(Y_pred == Y_val), len(Y_pred), (sum(Y_pred == Y_val) / len(Y_pred))))
    #print(data_region["gname"].value_counts())
    #print("\n")

regions = ["Australasia & Oceania",
             "Central America & Caribbean",
             "Central Asia",
             "East Asia",
             "Eastern Europe",
             "Middle East & North Africa",
             "North America",
             "South America",
             "South Asia",
             "Southeast Asia",
             "Sub-Saharan Africa",
             "Western Europe"]

results = pd.DataFrame(columns=('region', 'groups', 'accuracy'))
results_list = []
i = 0

for region in regions:
    for n_groups in range(50):
        model, accuracy, X_train, Y_train, X_val, Y_val, X_test, Y_test = get_accuracy(n_groups + 1, region)
        results.loc[i] = [region, n_groups + 1, accuracy]
        results_list.append({"model": model, "region": region, "n_groups": n_groups + 1, "X_train": X_train, 
                             "Y_train": Y_train, "X_val": X_val, "Y_val": Y_val, "X_test": X_test, "Y_test": Y_test})
        print("Did %s n%d" % (region, n_groups + 1))
        i = i + 1

results

plt.rcParams['figure.figsize']=(20,10)
ax = sns.pointplot(x="groups", y="accuracy", hue="region", data=results)

plt.rcParams['figure.figsize']=(20,10)
ax = sns.pointplot(x="groups", y="accuracy", hue="region", data=results.loc[(results["region"] != "Australasia & Oceania") &
                                                                           (results["region"] != "Eastern Europe") &
                                                                           (results["region"] != "Central Asia") &
                                                                           (results["region"] != "East Asia")])

data_test_final = [x for x in results_list if x["n_groups"] == 50]

for x in data_test_final:
    predicted_test = x["model"].predict(x["X_test"])
    real_test = x["Y_test"]
    predicted_val = x["model"].predict(x["X_val"])
    real_val = x["Y_val"]
    print("Accuracy for %s: val:%f, test:%f" % (x["region"], (sum(predicted_val == real_val) / len(real_val)), 
                                                             (sum(predicted_test == real_test) / len(real_test))))

data_with_unknown = pd.read_csv("terrorism_with_unknown_cleaned.csv")

data_with_unknown[data_with_unknown.gname == "Unknown"]["region_txt"].value_counts()

data_with_unknown = pd.get_dummies(data_with_unknown, columns = ["attacktype1_txt",
                     "targtype1_txt",
                     "weaptype1_txt",
                     "natlty1_txt",
                     "weaptype1_txt",
                     "weapsubtype1_txt"])

data_with_unknown[data_with_unknown.region_txt == "Middle East & North Africa"].gname.value_counts() / len(data_with_unknown[data_with_unknown.region_txt == "Middle East & North Africa"])

model = OneVsRestClassifier(RandomForestClassifier(random_state=2))

X_train = data_with_unknown.loc[(data_with_unknown.gname != "Unknown") & 
                                (data_with_unknown.region_txt == "Middle East & North Africa")].drop(["gname", "region_txt"], 
                                                                                                     axis=1)

Y_train = data_with_unknown.loc[(data_with_unknown.gname != "Unknown") & 
                                (data_with_unknown.region_txt == "Middle East & North Africa")].drop(["region_txt"], 
                                                                                                     axis=1)["gname"]

X_test = data_with_unknown.loc[(data_with_unknown.gname == "Unknown") & 
                                (data_with_unknown.region_txt == "Middle East & North Africa")].drop(["gname", "region_txt"], axis=1)

model.fit(X_train, Y_train)

predictions = model.predict(X_test)

predictions_series = pd.Series(predictions)



len(predictions)

predictions_series.value_counts() / len(predictions)



