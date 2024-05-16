import numpy as np
import matplotlib.pyplot as plt
import pickle, time, seaborn, random, json, os
get_ipython().magic('matplotlib inline')
from sklearn import tree
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

#Setting up the path in the working directory
Data_path = 'PLAID/'
csv_path = Data_path + 'CSV/'
csv_files = os.listdir(csv_path)

#Load meta data
with open(Data_path + 'meta1.json') as data_file:    
    meta1 = json.load(data_file)

meta = [meta1]

#Functions to parse meta data stored in JSON format
def clean_meta(ist):
    '''remove '' elements in Meta Data ''' 
    clean_ist = ist.copy()
    for k,v in ist.items():
        if len(v) == 0:
            del clean_ist[k]
    return clean_ist
                
def parse_meta(meta):
    '''parse meta data for easy access'''
    M = {}
    for m in meta:
        for app in m:
            M[int(app['id'])] = clean_meta(app['meta'])
    return M
            
Meta = parse_meta(meta)

# Unique appliance types
types = list(set([x['type'] for x in Meta.values()]))
types.sort()
#print(Unq_type)

def read_data_given_id_limited(path,ids,val,progress=False,last_offset=0):
    '''read data given a list of ids and CSV paths'''
    n = len(ids)
    if n == 0:
        return {}
    else:
        data = {}
        for (i,ist_id) in enumerate(ids, start=1):
            if last_offset==0:
                data[ist_id] = np.genfromtxt(path+str(ist_id)+'.csv',
                delimiter=',',names='current,voltage',dtype=(float,float))
            else:
                p=subprocess.Popen(['tail','-'+str(int(offset)),path+
                    str(ist_id)+'.csv'],stdout=subprocess.PIPE)
                data[ist_id] = np.genfromtxt(p.stdout,delimiter=',',
                    names='current,voltage',dtype=(float,float))
            data[ist_id]=data[ist_id][-val:]
           
        return data

#get all the data points
data={}

val=30000 # take only last 30,000 values as they are most likely to be in the steady state
ids_to_draw = {}

for (ii,t) in enumerate(Unq_type):
    t_ids = [i for i,j in enumerate(Types,start=1) if j == t]
    ids_to_draw[t] = t_ids
    data[t]=read_data_given_id_limited(csv_path, ids_to_draw[t], False,val)

# Saving or loading the main dictionary pickle file
saving = False
if saving:
    pickle_file = open('AppData.pkl','wb')
    pickle.dump(data,pickle_file,protocol=2)
    pickle_file.close()
else:
    pkf = open('AppData.pkl','rb')
    data = pickle.load(pkf)
    pkf.close()

#get house number and ids for each CSV
houses=[]
org_ids=[]

for i in range(0,len(Meta)):
    houses.append(Meta[i+1].get('location'))
    org_ids.append(i+1)
houses = np.hstack([np.array(houses)[:,None],np.array(org_ids)[:,None]])

cycle = 30000; num_cycles = 1; till = -cycle*num_cycles
resh = np.int(-till/num_cycles); tot = np.sum([len(data[x]) for x in data]); org_ids,c = [], 0
V = np.empty([resh,tot]); I = np.empty([resh,tot]); y = np.zeros(tot)
for ap_num,ap in enumerate(types):
    for i in data[ap]:
        V[:,c] = np.mean(np.reshape(data[ap][i]['voltage'][till:],(-1,cycle)),axis=0)
        I[:,c] = np.mean(np.reshape(data[ap][i]['current'][till:],(-1,cycle)),axis=0)
        y[c] = ap_num
        org_ids.append(i)
        c += 1
    pass
V_org = V.T; I_org = I.T

# plot V-I of last 10 steady state periods
num_figs = 5; fig, ax = plt.subplots(len(types),num_figs,figsize=(10,20)); till = -505*10
for (i,t) in enumerate(types):
    j = 0; p = random.sample(list(data[t].keys()),num_figs)
    for (k,v) in data[t].items():
        if j > num_figs-1:
            break
        if k not in p:
            continue
        ax[i,j].plot(v['current'][till:],v['voltage'][till:],linewidth=1)
        ax[i,j].set_title('Org_id: {}'.format(k),fontsize = 10); ax[i,j].set_xlabel('Current (A)',fontsize = 8) 
        ax[i,j].tick_params(axis='x', labelsize=5); ax[i,j].tick_params(axis='y', labelsize=8) 
        j += 1
    ax[i,0].set_ylabel('{} (V)'.format(t), fontsize=10)
fig.tight_layout()

saving = False
if saving:
    pickle_file = open('Data_matrices.pkl','wb')
    pickle.dump([V_org,I_org,y_org,org_ids,houses,types],pickle_file,protocol=2)
    pickle_file.close()
else:
    pkf = open('Data_matrices.pkl','rb')
    V_org,I_org,y_org,org_ids,houses,types = pickle.load(pkf)
    pkf.close()

cycle = 505; num_cycles = 1; till = -cycle*num_cycles
V = np.empty((V_org.shape[0],cycle)); I = np.empty((V_org.shape[0],cycle)); y = y_org; c = 0
for i,val in enumerate(V_org):
    V[i] = np.mean(np.reshape(V_org[i,till:],(-1,cycle)),axis=0)
    I[i] = np.mean(np.reshape(I_org[i,till:],(-1,cycle)),axis=0)

V = (V-np.mean(V,axis=1)[:,None]) / np.std(V,axis=1)[:,None]; I = (I-np.mean(I,axis=1)[:,None]) / np.std(I,axis=1)[:,None]

print_images = False; seaborn.reset_orig()
m = V.shape[0]; j = 0
temp = np.empty((m,32400)); p = random.sample(range(m),3)
for i in range(m):
    if print_images:
        fig = plt.figure(figsize=(2,2))
        plt.plot(I[i],V[i],linewidth=0.8,color='b'); plt.xlim([-4,4]); plt.ylim([-2,2]); 
        plt.savefig('pics_505_1/Ap_{}.png'.format(i))
        plt.close()
    else:
        im = Image.open('pics_505_1/Ap_{}.png'.format(i)).crop((20,0,200,200-20))
        im = im.convert('L')
        temp[i] = np.array(im).reshape((-1,))
        if i in p:
            display(im)
            j += 1
    pass
seaborn.set()
get_ipython().magic('matplotlib inline')

X = temp; y = y_org
X_, X_test, y_, y_test = train_test_split(X,y, test_size=0.2)
X_train, X_cv, y_train, y_cv = train_test_split(X_, y_, test_size=0.2)

def eval_cfls(models,X,y,X_te,y_te):
    ss = []; tt = []
    for m in models:
        start = time.time()
        m.fit(X,y)
        ss.append(np.round(m.score(X_te,y_te),4))
        print(str(m).split('(')[0],': {}'.format(ss[-1]),'...Time: {} s'.format(np.round(time.time()-start,3)))
        tt.append(np.round(time.time()-start,3))
    return ss,tt

models = [OneVsRestClassifier(LinearSVC(random_state=0)),tree.ExtraTreeClassifier(),tree.DecisionTreeClassifier(),GaussianNB(),
          BernoulliNB(),GradientBoostingClassifier(), KNeighborsClassifier(),RandomForestClassifier()]

ss,tt = eval_cfls(models,X_train,y_train,X_cv,y_cv)
rand_guess = np.random.randint(0,len(set(y_train)),size=y_cv.shape[0])
print('Random Guess: {}'.format(np.round(np.mean(rand_guess == y_cv),4)))

scores = []
for n in range(1,11,2):
    clf = KNeighborsClassifier(n_neighbors=n,weights='distance')
    clf.fit(X_train,y_train)
    scores.append(clf.score(X_cv, y_cv))
plt.plot(range(1,11,2),scores); plt.xlabel('Number of neighbors'); plt.ylabel('Accuracy'); plt.ylim([0.8,1]);
plt.title('K-nearest-neighbors classifier');

scores = []
for n in range(5,120,10):
    clf = RandomForestClassifier(n_estimators=n)
    clf.fit(X_train,y_train)
    scores.append(clf.score(X_cv, y_cv))
plt.plot(range(5,120,10),scores); plt.xlabel('Number of sub-trees'); plt.ylabel('Accuracy'); plt.ylim([0.8,1]);
plt.title('Random Forest classifier');

models = [KNeighborsClassifier(n_neighbors=1,weights='distance'),RandomForestClassifier(n_estimators=80)]
eval_cfls(models,np.vstack([X_train,X_cv]),np.hstack([y_train,y_cv]),X_test,y_test);

cv_scores = []; X = temp; y = y_org
p = np.random.permutation(X.shape[0])
X = X[p]; y = y[p];
for m in models:
    start = time.time()
    cv_scores.append(cross_val_score(m, X, y, cv=10))
    print(str(m).split('(')[0],'average score: {}'.format(np.round(np.mean(cv_scores),3)),
         '...10-fold CV Time: {} s'.format(np.round(time.time()-start,3)))

def held_house(name,houses):
    ids_te = houses[np.where(houses[:,0] == name),1].astype(int);
    ids_test,ids_train = [],[]
    for i,ID in enumerate(org_ids):
        if ID in ids_te:
            ids_test.append(i)
        else:
            ids_train.append(i)
    return ids_test,ids_train

X = temp; y = y_org; h_names = ['house{}'.format(i+1) for i in range(len(set(houses[:,0])))]
scores = np.zeros((len(h_names),2))
for i,m in enumerate(models):
    ss = []
    for h in h_names:
        ids_test,ids_train = held_house(h,houses)
        X_train, X_test = X[ids_train], X[ids_test]; 
        y_train,y_test = y[ids_train],y[ids_test];        
        m.fit(X_train,y_train)
        ss.append(m.score(X_test,y_test))
    
    scores[:,i] = np.array(ss)
    plt.figure(figsize = (12,3))
    plt.bar(np.arange(len(h_names)),scores[:,i],width=0.8); plt.xlim([0,len(h_names)]); plt.yticks(np.arange(0.1,1.1,0.1)); 
    plt.ylabel('Accuracy');
    plt.title('{} cross-validation per home. Median accuracy: {}'.format(str(m).split('(')[0],
                                                                         np.round(np.median(scores[:,i]),3)))
    plt.xticks(np.arange(len(h_names))+0.4,h_names,rotation='vertical');
plt.show()

df = pd.DataFrame(np.array([np.mean(scores,axis=0),np.sum(scores == 1,axis=0),
                   np.sum(scores >= 0.9,axis=0),np.sum(scores < 0.8,axis=0),np.sum(scores < 0.5,axis=0)]),columns=['KNN','RF'])
df['Stats'] = ['Avg. accuracy','100% accuracy','Above 90%','Above 80%','Below 50%']; 
df.set_index('Stats',inplace=True); df.head()

X = temp; y = y_org;
ids_test, ids_train = held_house('house46',houses)
X_train, X_test = X[ids_train], X[ids_test]; y_train,y_test = y[ids_train],y[ids_test]; 
V_,V_test = V[ids_train],V[ids_test]; I_,I_test = I[ids_train],I[ids_test]; org_ids_test = np.array(org_ids)[ids_test]
models[1].fit(X_train,y_train)
pred = models[1].predict(X_test)
items = np.where(pred != y_test)[0]
print('Number of wrong predictions in house13: {}'.format(len(items)))
for ids in items[:2]:
    print('Prediction: '+ types[int(pred[ids])],', Actual: '+types[int(y_test[ids])])
    fig,ax = plt.subplots(1,3,figsize=(11,3))
    ax[0].plot(I_test[ids],V_test[ids],linewidth=0.5); ax[0].set_title('Actual data. ID: {}'.format(org_ids_test[ids]));
    ax[1].plot(I_[y_train==y_test[ids]].T,V_[y_train==y_test[ids]].T,linewidth=0.5); 
    ax[1].set_title('Profiles of {}'.format(types[int(y_test[ids])]))
    ax[2].plot(I_[y_train==pred[ids]].T,V_[y_train==pred[ids]].T,linewidth=0.5); 
    ax[2].set_title('Profiles of {}'.format(types[int(pred[ids])])); 

def plot_clf_samples(model,X,X_te,y,y_te,n):
    model.fit(X[:n], y[:n])
    return np.array([model.score(X[:n], y[:n]), model.score(X_te, y_te)])

X = temp; y = y_org;
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
models[1].fit(X_train,y_train)
models[1].score(X_test, y_test)
nsamples = [int(x) for x in np.linspace(10, X_train.shape[0], 20)]
errors = np.array([plot_clf_samples(clf, X_train, X_test, y_train,y_test, n) for n in nsamples])
plt.plot(nsamples, errors[:,0], nsamples, errors[:,1]); plt.xlabel('Number of appliances'); plt.ylabel('Accuracy'); 
plt.ylim([0.4,1.1])
plt.legend(['Training accuracy','Test accuracy'],loc=4); plt.title('RF accuracy with respect of number of samples');

