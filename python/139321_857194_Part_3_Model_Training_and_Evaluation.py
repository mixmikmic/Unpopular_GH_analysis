import pandas as pd
import numpy as np
import os, warnings
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from modules.feature_extractor import (tokensToIds, countMatrix, priorProbabilityAnswer, posterioriProb, 
                               feature_selection, featureWeights, wordProbabilityInAnswer, 
                               wordProbabilityNotinAnswer, normalizeTF, getIDF, softmax)
from azureml.logging import get_azureml_logger
warnings.filterwarnings("ignore")

run_logger = get_azureml_logger()
run_logger.log('amlrealworld.QnA-matching.part3-model-training-eval','true')

workfolder = os.environ.get('AZUREML_NATIVE_SHARE_DIRECTORY')

# paths to trainQ and testQ.
trainQ_path = os.path.join(workfolder, 'trainQ_part2')
testQ_path = os.path.join(workfolder, 'testQ_part2')

# load the training and test data.
trainQ = pd.read_csv(trainQ_path, sep='\t', index_col='Id', encoding='latin1')
testQ = pd.read_csv(testQ_path, sep='\t', index_col='Id', encoding='latin1')

token2IdHashInit = tokensToIds(trainQ['Tokens'], featureHash=None)

# get unique answerId in ascending order
uniqueAnswerId = list(np.unique(trainQ['AnswerId']))

N_wQ = countMatrix(trainQ, token2IdHashInit)
idf = getIDF(N_wQ)

x_wTrain = normalizeTF(trainQ, token2IdHashInit)
x_wTest = normalizeTF(testQ, token2IdHashInit)

tfidfTrain = (x_wTrain.T * idf).T
tfidfTest = (x_wTest.T * idf).T

# calculate the count matrix of all training questions.
N_wAInit = countMatrix(trainQ, token2IdHashInit, 'AnswerId', uniqueAnswerId)

P_A = priorProbabilityAnswer(trainQ['AnswerId'], uniqueAnswerId)
P_Aw = posterioriProb(N_wAInit, P_A, uniqueAnswerId)

# select top N important tokens per answer class.
featureHash = feature_selection(P_Aw, token2IdHashInit, topN=19)
token2IdHash = tokensToIds(trainQ['Tokens'], featureHash=featureHash)

N_wA = countMatrix(trainQ, token2IdHash, 'AnswerId', uniqueAnswerId)

alpha = 0.0001
P_w = featureWeights(N_wA, alpha)

beta = 0.0001
P_wA = wordProbabilityInAnswer(N_wA, P_w, beta)
P_wNotA = wordProbabilityNotinAnswer(N_wA, P_w, beta)

NBWeights = np.log(P_wA / P_wNotA)

beta_A = 0

x_wTest = normalizeTF(testQ, token2IdHash)
Y_test_prob1 = softmax(-beta_A + np.dot(x_wTest.T, NBWeights))

X_train, Y_train = tfidfTrain.T, np.array(trainQ['AnswerId'])
clf = svm.LinearSVC(dual=True, multi_class='ovr', penalty='l2', C=1, loss="squared_hinge", random_state=1)
clf.fit(X_train, Y_train)

X_test = tfidfTest.T
Y_test_prob2 = softmax(clf.decision_function(X_test))

# train one-vs-rest classifier using NB scores as features.
def ovrClassifier(trainLabels, x_wTrain, x_wTest, NBWeights, clf, ratio):
    uniqueLabel = np.unique(trainLabels)
    dummyLabels = pd.get_dummies(trainLabels)
    numTest = x_wTest.shape[1]
    Y_test_prob = np.zeros(shape=(numTest, len(uniqueLabel)))

    for i in range(len(uniqueLabel)):
        X_train_all, Y_train_all = x_wTrain.T * NBWeights[:, i], dummyLabels.iloc[:, i]
        X_test = x_wTest.T * NBWeights[:, i]
        
        # with sample selection.
        if ratio is not None:
            # ratio = # of Negative/# of Positive
            posIdx = np.where(Y_train_all == 1)[0]
            negIdx = np.random.choice(np.where(Y_train_all == 0)[0], ratio*len(posIdx))
            allIdx = np.concatenate([posIdx, negIdx])
            X_train, Y_train = X_train_all[allIdx], Y_train_all.iloc[allIdx]
        else: # without sample selection.
            X_train, Y_train = X_train_all, Y_train_all
            
        clf.fit(X_train, Y_train)
        if hasattr(clf, "decision_function"):
            Y_test_prob[:, i] = clf.decision_function(X_test)
        else:
            Y_test_prob[:, i] = clf.predict_proba(X_test)[:, 1]

    return softmax(Y_test_prob)

x_wTrain = normalizeTF(trainQ, token2IdHash)
x_wTest = normalizeTF(testQ, token2IdHash)

clf = RandomForestClassifier(n_estimators=250, criterion='entropy', random_state=1)
Y_test_prob3 = ovrClassifier(trainQ["AnswerId"], x_wTrain, x_wTest, NBWeights, clf, ratio=3)

Y_test_prob_aggr = np.mean([Y_test_prob1, Y_test_prob2, Y_test_prob3], axis=0)

# get the rank of answerIds for a given question. 
def rank(frame, scores, uniqueAnswerId):
    frame['SortedAnswers'] = list(np.array(uniqueAnswerId)[np.argsort(-scores, axis=1)])
    
    rankList = []
    for i in range(len(frame)):
        rankList.append(np.where(frame['SortedAnswers'].iloc[i] == frame['AnswerId'].iloc[i])[0][0] + 1)
    frame['Rank'] = rankList
    
    return frame

testQ = rank(testQ, Y_test_prob_aggr, uniqueAnswerId)

AR = np.floor(testQ['Rank'].mean())
top3 = round(len(testQ.query('Rank <= 3'))/len(testQ), 3)
 
print('Average of rank: ' + str(AR))
print('Percentage of questions find answers in the first 3 choices: ' + str(top3))

