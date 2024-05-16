import skimage.measure as sk
from skimage import io, color
import matplotlib.pyplot as plt

path = "/Users/sreejithmenon/Dropbox/Social_Media_Wildlife_Census/All_Zebra_Count_Images/"

img1 = io.imread(path+"167.jpeg")
img2 = io.imread(path+"168.jpeg")

# STRUCTURAL SIMILARITY; higher means similar 

IMG1 = color.rgb2gray(img1)
IMG2 = color.rgb2gray(img2)
print(sk.compare_ssim(IMG1, IMG2))
 

## SQUARED ERRORS; lower means similar

# difference in colored images
print(sk.simple_metrics.compare_nrmse(img1, img2))
print(sk.simple_metrics.compare_mse(img1, img2))

# difference in gray scale images
print(sk.simple_metrics.compare_nrmse(IMG1, IMG2))
print(sk.simple_metrics.compare_mse(IMG1, IMG2))

fig = plt.figure("Images")

ax = fig.add_subplot(1,2,1)
plt.imshow(img1, cmap=plt.cm.gray)

ax = fig.add_subplot(1,2,2)
plt.imshow(img2, cmap=plt.cm.gray)
plt.show()

import PopulationEstimatorAPI as PE, ClassiferHelperAPI as CH

regrArgs = {'linear' : {'fit_intercept' : True},
            'ridge' : {'fit_intercept' : True},
            'lasso' : {'fit_intercept' : True},
            'elastic_net' : {'fit_intercept' : True},
            'svr' : {'fit_intercept' : True},
            'dtree_regressor' : {'fit_intercept' : True}}

train_fl = "../data/BeautyFtrVector_GZC_Expt2.csv"
test_fl = "../data/GZC_exifs_beauty_full.csv"

methObj,predResults = CH.trainTestRgrs(train_fl,
                                test_fl,
                                'linear',
                                'beauty',
                                infoGainFl="../data/infoGainsExpt2.csv",
                                methArgs = regrArgs
                                )

predResults['1'], predResults['2']

import pandas as pd, numpy as np
import ClassifierCapsuleClass as ClfClass, ClassiferHelperAPI as CH
clfArgs = {'dummy' : {'strategy' : 'most_frequent'},
            'bayesian' : {'fit_prior' : True},
            'logistic' : {'penalty' : 'l2'},
            'svm' : {'kernel' : 'rbf','probability' : True},
            'dtree' : {'criterion' : 'entropy'},
            'random_forests' : {'n_estimators' : 10 },
            'ada_boost' : {'n_estimators' : 50 }}

methodName = 'logistic'

train_data_fl = "../data/BeautyFtrVector_GZC_Expt2.csv"
train_x = pd.DataFrame.from_csv(train_data_fl)
        
train_x = train_x[(train_x['Proportion'] >= 80.0) | (train_x['Proportion'] <= 20.0)]
train_x['TARGET'] = np.where(train_x['Proportion'] >= 80.0, 1, 0)

train_y = train_x['TARGET']
train_x.drop(['Proportion','TARGET'],1,inplace=True)        
clf = CH.getLearningAlgo(methodName,clfArgs.get(methodName,None))
lObj = ClfClass.ClassifierCapsule(clf,methodName,0.0,train_x,train_y,None,None)

test_data_fl = "../data/GZC_exifs_beauty_full.csv"
testDf = pd.DataFrame.from_csv(test_data_fl)

testDataFeatures = testDf[lObj.train_x.columns]

with open("../data/HumanImagesException.csv", "r") as HImgs:
    h_img_list = HImgs.read().split("\n")

h_img_list = list(map(int, h_img_list))

len(set(testDataFeatures.index) - set(h_img_list))

count = 0
for i in h_img_list:
    if i in testDataFeatures.index:
        count += 1
print(count)

len(testDf)

testDataFeatures.index = set(testDataFeatures.index) - set(h_img_list)

obj

