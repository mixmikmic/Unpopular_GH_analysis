import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

sources=np.load('../data/sources_1layer.npy')
data=np.load('../data/data_1layer.npy')
V=np.load('../data/V_1layer.npy')
vmin=1500
vmax=5000
# Normalize the velocities as it tends to make training easier
V=(V-vmin)/(vmax-vmin)
print(sources.shape, data.shape, V.shape)

# Include the source wavelet with the data
X = np.hstack([sources, data])
print(X.shape)

# Set aside 100 samples for testing
ntrain = X.shape[0]-100
X_train = X[:ntrain,:]
V_train = V[:ntrain,:]
X_test = X[ntrain:,:]
V_test = V[ntrain:,:]

plt.plot(X[4,:])

plt.figure(figsize=(12,12))
plt.imshow(X[:250,:], aspect='auto')
plt.figure(figsize=(12,12))
plt.imshow(V[:250,:], aspect='auto')

num_hidden = 250
init=True
if init:
    tf.reset_default_graph()

    X_tf = tf.placeholder(tf.float32, (None, X.shape[1]))
    V_tf = tf.placeholder(tf.float32, (None, V.shape[1]))

    l1 = tf.layers.dense(X_tf, num_hidden, activation=tf.nn.relu, name='l1')
    l2 = tf.layers.dense(l1, num_hidden, activation=tf.nn.relu, name='l2')
    l3 = tf.layers.dense(l2, num_hidden, activation=tf.nn.relu, name='l3')
    l4 = tf.layers.dense(l3, num_hidden, activation=tf.nn.relu, name='l4')
    l5 = tf.layers.dense(l4, num_hidden, activation=tf.nn.relu, name='l5')
    l6 = tf.layers.dense(l5, num_hidden, activation=tf.nn.relu, name='l6')
    l7 = tf.layers.dense(l6, num_hidden, activation=tf.nn.relu, name='l7')
    l8 = tf.layers.dense(l7, V.shape[1], name='l8')

    loss = tf.losses.mean_squared_error(V_tf, l8)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    batch_size=50

    def test_loss():
        return sess.run(loss, feed_dict={X_tf: X_test, V_tf: V_test})

    def test_prediction():
        return sess.run(l8, feed_dict={X_tf: X_test, V_tf: V_test})


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

for step in range(20000):
    bs = step*batch_size %(X_train.shape[1] - batch_size)
    x_batch = X_train[bs:bs+batch_size, :]
    v_batch = V_train[bs:bs+batch_size, :]
    _, l = sess.run([train_op, loss], feed_dict={X_tf: x_batch, V_tf: v_batch})
    if step % 1000 == 0:
        print(step, l, test_loss())
print(test_loss())

for i in [4,5]:
    plt.figure()
    plt.plot(test_prediction()[i,:])
    plt.plot(V_test[i,:])

plt.imshow(test_prediction(), aspect='auto', vmin=0, vmax=1)
plt.figure()
plt.imshow(V_test, aspect='auto', vmin=0, vmax=1)

