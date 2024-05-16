import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

import sys, os
sys.path.append(os.pardir)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784], name = 'x_')
y = tf.placeholder('float', name = 'y_')

hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1]), name = 'w1'),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]), name = 'b1')}

hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]), name = 'w2'),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]), name = 'b2')}

hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]), name = 'w3'),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]), name = 'b3')}

output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes]), name = 'w4'),
                    'biases':tf.Variable(tf.random_normal([n_classes]), name = 'b4')}

sess=tf.Session()
saver = tf.train.Saver()

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'],name = 'l1_')
    l1 = tf.nn.relu(l1, name = 'l1')

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'],name = 'l2_')
    l2 = tf.nn.relu(l2, name = 'l2')

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'],name = 'l3_')
    l3 = tf.nn.relu(l3, name = 'l3')

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

save_path = 'pyhelp/'
model_name = 'sy'
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path_full = os.path.join(save_path, model_name)

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
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

