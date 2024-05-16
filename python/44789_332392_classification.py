import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('input/data.csv')

dataframe = dataframe.drop(['index', 'price', 'sq_price'],axis=1)
dataframe = dataframe[0:10]
print (dataframe)

# Add labels
# 1 is good buy and 0 is bad buy
dataframe.loc[:, ('y1')]  = [1,0,1,1,0,1,1,0,0,1]

#y2 is the negation of the y1
dataframe.loc[:, ('y2')] = dataframe['y1'] == 0

# convert true/false value to 1s and 0s
dataframe.loc[:, ('y2')] = dataframe['y2'].astype(int)
dataframe

inputX = dataframe.loc[:, ['area','bathrooms']].as_matrix()
inputY = dataframe.loc[:, ['y1','y2']].as_matrix()
inputX

# Hyperparameters
learning_rate = 0.00001
training_epochs = 2000 #iterations
display_steps = 50
n_samples = inputY.size

# Create our computation graph/ Neural Network
# for feature input tensors, none means any numbers of examples
# placgeholder are gateways for data into our computation
x = tf.placeholder(tf.float32, [None,2])

#create weights
# 2 X 2 float matrix, that we'll keep updateing through the training process
w = tf.Variable(tf.zeros([2,2]))

#create bias , we have 2 bias since we have two features
b = tf.Variable(tf.zeros([2]))

y_values = tf.add(tf.matmul(x,w),b)

y = tf.nn.softmax(y_values)

# For trainign purpose, we'll also feed a matrix of labels
y_ = tf.placeholder(tf.float32, [None,2]) 

#Cost function: Mean squared error
cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)
# Gradient Descent 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Initialize variables and tensorflow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Now for the actual training 
for i in range(training_epochs):
    #Take a gradient descent step using our input and labels 
    sess.run(optimizer, feed_dict={x: inputX, y_:inputY})
     
    # Display logs per epoch step   
    if (i)  % display_steps == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_: inputY})
        print("Training step:", '%04d' % (i), "Cost: :", "{:.9f}".format(cc) )
    
print("Optimization Done!")
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print ("Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n')

sess.run(y, feed_dict={x:inputX})

