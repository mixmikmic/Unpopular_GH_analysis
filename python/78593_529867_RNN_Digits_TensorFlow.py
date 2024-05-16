import tensorflow as tf
# loading the data

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# import rnn from tensorFlow
from tensorflow.contrib import rnn 

# define number of classes = 10 for digits 0 through 9 
n_classes = 10

# defining the chunk size, number of chunks, and rnn size as new variables
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

# placeholders for variables x and y
x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):

    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

hm_epochs = 4

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    #optimizer is learning rate but in this casde the default s fine
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            #_ is variable we dont care about
            #we have total number items and divide by batch_size for number of batches
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                #c is cost
                _, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
                epoch_loss += c

            print('Epoch', epoch, ' completed out of ', hm_epochs, ' loss: ', epoch_loss)

        #we come here after optimizing the weights
        #tf.argmax is going to return the index of max values
        #in this training module we are checking if the prediction is matching the value
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        #accuracy is the float value of correct
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #this is just evaluation
        print('Accuracy: ', accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)

