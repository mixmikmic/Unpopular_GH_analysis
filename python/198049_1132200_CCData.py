import pandas as pd
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from sklearn.metrics import auc, accuracy_score, roc_auc_score

data = pd.read_excel('ccdata.xls', header = 1)

data.head()

data.drop('ID', axis = 1, inplace = True)

# Check for null values.
data.isnull().sum().sort_values(ascending=False)

X = data.drop(['default payment next month'], axis=1)
y = data['default payment next month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

estimator = lgb.LGBMClassifier(learning_rate = 0.125, metric = 'l1', 
                        n_estimators = 20, num_leaves = 38)

param_grid = {
    'n_estimators': [x for x in range(20, 36, 2)],
    'learning_rate': [0.10, 0.125, 0.15, 0.175, 0.2]}
gridsearch = GridSearchCV(estimator, param_grid)

gridsearch.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['auc', 'binary_logloss'],
early_stopping_rounds=5)


print('Best parameters found by grid search are:', gridsearch.best_params_)


gbm = lgb.LGBMClassifier(learning_rate = 0.125, metric = 'l1', 
                        n_estimators = 20, num_leaves = 38)


gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['auc', 'binary_logloss'],
early_stopping_rounds=5)

ax = lgb.plot_importance(gbm, height = 0.4, max_num_features=25, xlim = (0,100), ylim = (0,23), 
                         figsize = (10,6))
plt.show()

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
print('The accuracy of prediction is:', accuracy_score(y_test, y_pred))
print('The roc_auc_score of prediction is:', roc_auc_score(y_test, y_pred))
print('The null acccuracy is:', max(y_test.mean(), 1 - y_test.mean()))

from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_pred))

y_pred_prob = gbm.predict_proba(X_test)[:, 1]

y_pred_prob

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.grid(True)

print(metrics.roc_auc_score(y_test, y_pred_prob))



