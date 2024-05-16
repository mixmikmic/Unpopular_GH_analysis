import tensorflow as tf
import glob
from itertools import groupby
from collections import defaultdict

sess = tf.InteractiveSession()
image_filenames = glob.glob("./dataset/StanfordDogs/n02*/*.jpg")
image_filenames[0:2]
training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)
image_filename_with_breed = map(lambda filename: (filename.split("/")[2], 
                                                 filename), image_filenames)
for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
    for i, breed_image in enumerate(breed_images):
        if i % 5 == 0:
            testing_dataset[dog_breed].append(breed_image[1])
        else:
            training_dataset[dog_breed].append(breed_image[1])
    breed_training_count = len(training_dataset[dog_breed])
    breed_testing_count = len(testing_dataset[dog_breed])
    breed_training_count_float = float(breed_training_count)
    breed_testing_count_float = float(breed_testing_count)
    assert round(breed_testing_count_float / (breed_training_count_float +         breed_testing_count_float), 2) > 0.18, "Not enough testing images."
print("------------training_dataset testing_dataset END --------------------")
print(len(testing_dataset))
print(len(training_dataset))

def write_records_file(dataset, record_location):
    writer = None
    current_index = 0
    for breed, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if current_index % 100 == 0:
                if writer:
                    writer.close()
                record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=record_location,
                    current_index=current_index)
                writer = tf.python_io.TFRecordWriter(record_filename)
                print ("----------------------"+record_filename + "---------------------------") 
            current_index += 1
            image_file = tf.read_file(image_filename)
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print(image_filename)
                continue
            grayscale_image = tf.image.rgb_to_grayscale(image)
            resized_image = tf.image.resize_images(grayscale_image, [250, 151])
            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            image_label = breed.encode("utf-8")
            example = tf.train.Example(features=tf.train.Features(feature={
              'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
              'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))
            writer.write(example.SerializeToString())
    #writer.close()

#write_records_file(testing_dataset, "./result/test/testing-image")
write_records_file(training_dataset, "./result/train/training-image")
print("------------------write_records_file testing_dataset training_dataset END-------------------")
filename_queue = tf.train.string_input_producer(
tf.train.match_filenames_once("./result/test/*.tfrecords"))

reader = tf.TFRecordReader()
_, serialized = reader.read(filename_queue)
features = tf.parse_single_example(
serialized,
    features={
        'label': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
    })
record_image = tf.decode_raw(features['image'], tf.uint8)
image = tf.reshape(record_image, [250, 151, 1])
label = tf.cast(features['label'], tf.string)
min_after_dequeue = 10
batch_size = 3
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
print("---------------------load image from TFRecord END----------------------")

float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)
conv2d_layer_one = tf.contrib.layers.convolution2d(
    float_image_batch,
    num_outputs=32,
    kernel_size=(5,5),
    activation_fn=tf.nn.relu,
    weights_initializer=tf.contrib.layers.xavier_initializer(),
    stride=(2, 2),
    trainable=True)
pool_layer_one = tf.nn.max_pool(conv2d_layer_one,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
conv2d_layer_one.get_shape(), pool_layer_one.get_shape()
print("--------------------------------conv2d_layer_one pool_layer_one END--------------------------------")
conv2d_layer_two = tf.contrib.layers.convolution2d(
    pool_layer_one,
    num_outputs=64,
    kernel_size=(5,5),
    activation_fn=tf.nn.relu,
    weights_initializer=tf.contrib.layers.xavier_initializer(),
    stride=(1, 1),
    trainable=True)
pool_layer_two = tf.nn.max_pool(conv2d_layer_two,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
conv2d_layer_two.get_shape(), pool_layer_two.get_shape()
print("-----------------------------conv2d_layer_two pool_layer_two END---------------------------------")

flattened_layer_two = tf.reshape(pool_layer_two, [batch_size, -1])
flattened_layer_two.get_shape()
print("----------------------------------------flattened_layer_two END-----------------------------------------")
hidden_layer_three = tf.contrib.layers.fully_connected(
    flattened_layer_two, 512,
    weights_initializer=lambda i, dtype, partition_info=None: tf.truncated_normal([38912, 512], stddev=0.1),
    activation_fn=tf.nn.relu)
hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.1)
final_fully_connected = tf.contrib.layers.fully_connected(
    hidden_layer_three,
    120,
    weights_initializer=lambda i, dtype, partition_info=None: tf.truncated_normal([512, 120], stddev=0.1))
print("-----------------------final_fully_connected END--------------------------------------")

labels = list(map(lambda c: c.split("/")[-1], glob.glob("./dataset/StanfordDogs/*")))
train_labels = tf.map_fn(lambda l: tf.where(tf.equal(labels, l))[0,0:1][0], label_batch, dtype=tf.int64)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=final_fully_connected, labels=train_labels))
batch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.01, batch * 3, 120, 0.95, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
train_prediction = tf.nn.softmax(final_fully_connected)
print(train_prediction)
print("--------------------------------train_prediction END---------------------------------------")
filename_queue.close(cancel_pending_enqueues=True)
print("-------------------------------END---------------------------")



