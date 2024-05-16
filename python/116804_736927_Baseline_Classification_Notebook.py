get_ipython().magic('matplotlib inline')
import pickle
get_ipython().magic('run helper_loans.py')
pd.options.display.max_columns = 1000
plt.rcParams["figure.figsize"] = (15,10)
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier

df = unpickle_object("clean dataframe.pkl")

df.shape

y = df['loan_status_Late'].values
df.drop('loan_status_Late', inplace=True, axis=1)
X = df.values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
# params = {'strategy': ["stratified", "most_frequent", "prior", "uniform", "constant"]}
model = DummyClassifier(strategy = "stratified", random_state=0)
model.fit(X_train,y_train)
model.score(X_test,y_test)

model2 = DummyClassifier(strategy = "most_frequent", random_state=0)
model2.fit(X_train,y_train)
model2.score(X_test,y_test)

model3 = DummyClassifier(strategy = "prior", random_state=0)
model3.fit(X_train,y_train)
model3.score(X_test,y_test)

model4 = DummyClassifier(strategy = "uniform", random_state=0)
model4.fit(X_train,y_train)
model4.score(X_test,y_test)

model5 = DummyClassifier(strategy = "constant", random_state=0, constant=0)
model5.fit(X_train,y_train)
model5.score(X_test,y_test)

