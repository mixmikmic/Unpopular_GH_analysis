import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
import os
import cv2

n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, 784], name = 'x_')
y = tf.placeholder('float', name = 'y_')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32, name = 'prob')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name='conv2d')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME' , name = 'maxpool2d')

weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32]), name = 'w_c1'),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64]), name = 'w_c2'),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024]), name = 'w_f'),
               'out':tf.Variable(tf.random_normal([1024, n_classes]), name ='w_out')}

biases = {'b_conv1':tf.Variable(tf.random_normal([32]), name = 'b_c1'),
               'b_conv2':tf.Variable(tf.random_normal([64]), name = 'b_c2'),
               'b_fc':tf.Variable(tf.random_normal([1024]), name = 'b_f'),
               'out':tf.Variable(tf.random_normal([n_classes]), name = 'b_out')}

sess=tf.Session()
saver = tf.train.Saver()

def convolutional_neural_network(x):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'] , name = 'c1_')
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'], name = 'c2_')
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'], name = 'fc_')
    fc = tf.nn.dropout(fc, keep_rate, name = 'fc')

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

save_path = 'pyhelp/'
model_name = 'sy2'
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path_full = os.path.join(save_path, model_name)

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 30
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
    
        saver.save(sess,save_path_full)

train_neural_network(x)

