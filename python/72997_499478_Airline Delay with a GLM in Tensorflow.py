get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

df = pd.read_csv("2008.csv")
df.shape[0]

df = df[0:1000000]

df.columns

df[0:5]

df = pd.concat([df, pd.get_dummies(df["Origin"], prefix="Origin")], axis=1);
df = pd.concat([df, pd.get_dummies(df["Dest"  ], prefix="Dest"  )], axis=1);
df = df.dropna(subset=["ArrDelay"]) 
df["IsArrDelayed" ] = (df["ArrDelay"]>0).astype(int)
df[0:5]

train = df.sample(frac=0.8)
test  = df.drop(train.index)

#get the list of one hot encoding columns
OriginFeatCols = [col for col in df.columns if ("Origin_" in col)]
DestFeatCols   = [col for col in df.columns if ("Dest_"   in col)]
features = train[["Year","Month",  "DayofMonth" ,"DayOfWeek", "DepTime", "AirTime", "Distance"] + OriginFeatCols + DestFeatCols  ]
#features = train[["DepTime", "AirTime", "Distance"]]
labels   = train["IsArrDelayed"]

#convert it to numpy array to feed in tensorflow
featuresMatrix = features.as_matrix()
labelsMatrix   = labels  .as_matrix().reshape(-1,1)

features.shape[0]

featureSize = features.shape[1]
labelSize   = 1

training_epochs = 25
batch_size = 2500

graph = tf.Graph()
with graph.as_default():   
    # tf Graph Input
    LR = tf.placeholder(tf.float32 , name = 'LearningRate')
    X = tf.placeholder(tf.float32, [None, featureSize], name="features") # features
    Y = tf.placeholder(tf.float32, [None, labelSize], name="labels")   # training label

    with tf.name_scope("model") as scope:    
        # Set model weights
        W = tf.Variable(tf.random_normal([featureSize, labelSize],stddev=0.001), name="coefficients")
        B = tf.Variable(tf.random_normal([labelSize], stddev=0.001), name="bias")
        
        # Construct model
        logits = tf.matmul(X, W) + B                        
        with tf.name_scope("prediction") as scope:    
            P      = tf.nn.sigmoid(logits)

    with tf.name_scope("loss") as scope:
        with tf.name_scope("L2") as scope:             
           L2  = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
                
        # Minimize error using cross entropy
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets=Y, logits=logits) ) + 1E-5*L2
            
    with tf.name_scope("optimizer") as scope:
        # Gradient Descent
        optimizer = tf.train.AdamOptimizer(LR).minimize(cost)

    #used to make training plot on tensorboard
    with tf.name_scope("summary") as scope:
        tf.scalar_summary('cost', cost)
        tf.scalar_summary('L2', L2)
        SUMMARY = tf.merge_all_summaries()
    
        
    # Initializing the variables
    init = tf.initialize_all_variables()
      
sess = tf.Session(graph=graph)
tfTrainWriter = tf.train.SummaryWriter("./tfsummary/train", graph)        
sess.run(init)

# Training cycle
avg_cost_prev = -1
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(features.shape[0]/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = featuresMatrix[i*batch_size:(i+1)*batch_size]#features[i*batch_size:(i+1)*batch_size].as_matrix()
        batch_ys = labelsMatrix[i*batch_size:(i+1)*batch_size]#labels  [i*batch_size:(i+1)*batch_size].as_matrix().reshape(-1,1)

        #set learning rate
        learning_rate = 0.1 * pow(0.2, (epoch + float(i)/total_batch))
        
        # Fit training using batch data
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, LR:learning_rate})

        # Compute average loss
        avg_cost += c / total_batch
        
        #uncomment to send tensorflow summaries to tensorboard
        #summary = sess.run([SUMMARY], feed_dict={X: batch_xs, Y: batch_ys, LR:learning_rate})
        #tfTrainWriter.add_summary(summary, (epoch + float(i)/total_batch))
        
    # Display logs per epoch step
    print("Epoch: %04d, LearningRate=%.9f, cost=%.9f" % (epoch+1, learning_rate, avg_cost) )
               
    #check for early stopping
    if(avg_cost_prev>=0 and (abs(avg_cost-avg_cost_prev))<1e-4):
        print("early stopping")
        break
    else: avg_cost_prev = avg_cost        
print("Optimization Finished!")

w = sess.run(W, feed_dict={X: batch_xs, Y: batch_ys, LR:learning_rate})
coef = pd.DataFrame(data=w, index=features.columns, columns=["Coef"])
coef = coef.reindex( coef["Coef"].abs().sort_values(axis=0,ascending=False).index )  #order by absolute coefficient magnitude
coef[ coef["Coef"].abs()>0 ] #keep only non-null coefficients
coef[ 0:10 ] #keep only the 10 most important coefficients

testFeature = test[["Year","Month",  "DayofMonth" ,"DayOfWeek", "DepTime", "AirTime", "Distance"] + OriginFeatCols + DestFeatCols  ]
pred = sess.run(P, feed_dict={X: testFeature.as_matrix()})
test["IsArrDelayedPred"] = pred
test[0:10]

fpr, tpr, _ = roc_curve(test["IsArrDelayed"], test["IsArrDelayedPred"])
AUC = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=4, label='ROC curve (area = %0.3f)' % AUC)
plt.legend(loc=4)

AUC



