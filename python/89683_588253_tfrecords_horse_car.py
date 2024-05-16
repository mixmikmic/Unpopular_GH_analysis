import os 
import tensorflow as tf 
from PIL import Image  
import matplotlib.pyplot as plt 

cwd=os.getcwd()
classes={'car','horse'} 
writer= tf.python_io.TFRecordWriter("car_horse.tfrecords") 

for index,name in enumerate(classes):
    class_path=cwd+'/'+name+'/'
    for img_name in os.listdir(class_path): 
        img_path=class_path+img_name 
        img=Image.open(img_path)
        img= img.resize((128,128))
        img_raw=img.tobytes()
        #plt.imshow(img) # if you want to check you image,please delete '#'
        #plt.show()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        })) 
        writer.write(example.SerializeToString()) 

writer.close()

def read_and_decode(filename): # read iris_contact.tfrecords
    filename_queue = tf.train.string_input_producer([filename])# create a queue

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
               features={'label': tf.FixedLenFeature([], tf.int64),
               'img_raw' : tf.FixedLenFeature([], tf.string),})#return image and label

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])  #reshape image to 512*80*3
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #throw img tensor
    label = tf.cast(features['label'], tf.int32) #throw label tensor
    return img, label

filename_queue = tf.train.string_input_producer(["car_horse.tfrecords"]) 
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #return file and file_name
features = tf.parse_single_example(serialized_example,
           features={'label': tf.FixedLenFeature([], tf.int64),
                     'img_raw' : tf.FixedLenFeature([], tf.string),})  
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [128, 128, 3])
label = tf.cast(features['label'], tf.int32)
with tf.Session() as sess: 
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(5):
        example, l = sess.run([image,label])#take out image and label
        img=Image.fromarray(example, 'RGB')
        img.save(cwd+str(i)+'_Label_'+str(l)+'.jpg')#save image
        print(example, l, example.shape)
    coord.request_stop()
    coord.join(threads)

