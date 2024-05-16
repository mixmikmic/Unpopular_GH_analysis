get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn import metrics
import matplotlib.pyplot as plt

df_pima = pd.read_csv('C:/Users/nmannheimer/PycharmProjects/Code Projects/Machine Learning/pima-indians-diabetes.csv',
                      names=('Number of times pregnant',
                      'glucose tolerance test',
                      'Diastolic blood pressure mm Hg',
                      'Triceps skin fold thickness',
                      '2-Hour serum insulin mu U/ml',
                      'BMI',
                      'Diabetes pedigree function',
                      'Age',
                      'Class'))

print df_pima.info()

df_pima.describe()

df_pima['is_train'] = np.random.uniform(0, 1, len(df_pima)) <= 0.75
train = df_pima[df_pima['is_train'] == True]
test = df_pima[df_pima['is_train'] == False]

trainTargets = np.array(train['Class']).astype(int)
testTargets = np.array(test['Class']).astype(int)
features = df_pima.columns[0:8]

model = RandomForestClassifier()
predictions = model.fit(train[features], trainTargets).predict(test[features])

results = np.array(predictions)
scoring = np.array(testTargets)

accuracy = metrics.accuracy_score(testTargets, predictions)
print "The model produced {0}% accurate predictions.".format(accuracy*100)
print " "

y_true = testTargets
y_pred = results
print(classification_report(y_true, y_pred))

print 'What is the importance of each feature?'
for feat in zip(features,model.feature_importances_):
    print feat

y_pos = np.arange(len(features))
plt.bar(y_pos,model.feature_importances_, align='center',alpha=0.5)
plt.xticks(y_pos, features)
plt.title('Pima Feature Importances')
plt.xticks(rotation=90)
plt.show()

joblib.dump(model, 'C:/Users/nmannheimer/Desktop/DataScience/TabPy Training/Completed Models/JupyterPimaForest.pkl')

