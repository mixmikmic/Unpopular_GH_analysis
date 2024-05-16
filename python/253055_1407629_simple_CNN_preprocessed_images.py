import os
import sys
import pandas as pd
import numpy as np
import PIL

seed = 16
np.random.seed(seed)

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization

import keras.utils
from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

#check using system GPU for processing and declaring system/GPU parameters

from tensorflow.python.client import device_lib
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
print(device_lib.list_local_devices())

# configure tensorflow before fitting model
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.Session(config=tf_config)

# changing directory to access data (as numpy arrays)
os.chdir('C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets')

# define functions to load data

def load_array(fname):
    return np.load(open(fname,'rb'))

# load in labels and data (as tensors)

train_labels=load_array('train_labels.npy')
valid_labels=load_array('valid_labels.npy')

train_tensor=load_array('train_dataset.npy')

def Normalize_Input(X):
    minimum=0
    maximum=255
    X-minimum/(maximum-minimum)
    return X  

train_tensor=Normalize_Input(train_tensor)

valid_tensor=load_array('valid_dataset.npy')

valid_tensor=Normalize_Input(valid_tensor)

# feeding the training data through an Image Augmentation process (including resizing and shifting tolerance)

num_classes = 120
batch_size = 12
input_shape = (224, 224, 3)

datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, 
                             zoom_range=0.1, horizontal_flip=True)


# note to self... perhaps the imagedatagenerator parameters I had before were root cause of low accuracy...


train_generator = datagen.flow(x=train_tensor, y=train_labels, batch_size=batch_size)
validation_generator = datagen.flow(x=valid_tensor, y=valid_labels, batch_size=batch_size)

# fit the ImageDataGenerator 
datagen.fit(train_tensor)

wide_model_slow_learn = Sequential()

wide_model_slow_learn.add(BatchNormalization(input_shape=input_shape))
wide_model_slow_learn.add(Conv2D(64, (3, 3), strides=1, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_slow_learn.add(BatchNormalization())

wide_model_slow_learn.add(Conv2D(64, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_slow_learn.add(BatchNormalization())

wide_model_slow_learn.add(Conv2D(64, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_slow_learn.add(BatchNormalization())

wide_model_slow_learn.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_slow_learn.add(BatchNormalization())

wide_model_slow_learn.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_slow_learn.add(BatchNormalization())

wide_model_slow_learn.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_slow_learn.add(BatchNormalization())

wide_model_slow_learn.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(Dropout(0.2))
wide_model_slow_learn.add(GlobalAveragePooling2D())

wide_model_slow_learn.add(Dense(num_classes, activation='softmax'))
    
# Compile model

adam_op = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
wide_model_slow_learn.compile(loss='sparse_categorical_crossentropy', optimizer=adam_op, metrics=['accuracy']) 
#loss changed to sparse for new label data
print(wide_model_slow_learn.summary())

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='saved_models/weights.bestaugmented.from_scratch_v3.hdf5', 
                               verbose=1, save_best_only=True)

wide_model_slow_learn.fit_generator(train_generator, validation_data=validation_generator,
                         steps_per_epoch=800, epochs=10, callbacks=[checkpointer, early_stopping])

# try faster learning rate, see if we can speed up the improvement

wide_model_fast_learn = Sequential()

wide_model_fast_learn.add(BatchNormalization(input_shape=input_shape))
wide_model_fast_learn.add(Conv2D(64, (3, 3), strides=1, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
wide_model_fast_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_fast_learn.add(BatchNormalization())

wide_model_fast_learn.add(Conv2D(64, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_fast_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_fast_learn.add(BatchNormalization())

wide_model_fast_learn.add(Conv2D(64, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_fast_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_fast_learn.add(BatchNormalization())

wide_model_fast_learn.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_fast_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_fast_learn.add(BatchNormalization())

wide_model_fast_learn.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_fast_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_fast_learn.add(BatchNormalization())

wide_model_fast_learn.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_fast_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_fast_learn.add(BatchNormalization())

wide_model_fast_learn.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3)))
wide_model_fast_learn.add(Dropout(0.2))
wide_model_fast_learn.add(GlobalAveragePooling2D())

wide_model_fast_learn.add(Dense(num_classes, activation='softmax'))
    
# Compile model

adam_op = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
wide_model_fast_learn.compile(loss='sparse_categorical_crossentropy', optimizer=adam_op, metrics=['accuracy']) 
#loss changed to sparse for new label data
print(wide_model_fast_learn.summary())

checkpointer = ModelCheckpoint(filepath='saved_models/weights.bestaugmented.from_scratch_wide_model_fast_learn_v2.hdf5', 
                               verbose=1, save_best_only=True)

history_wmfl = wide_model_fast_learn.fit_generator(train_generator, validation_data=validation_generator,
                         steps_per_epoch=800, epochs=20, callbacks=[checkpointer, early_stopping])

wide_model_fast_learn.save('saved_models/wide_model_fast_learn.h5')

# lets plot/visualize the model training progress

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('bmh')

font = {'family' : 'sans-serif',
        'weight' : 'medium',
        'size'   : 16}

plt.rc('font', **font)

print(history_wmfl.history.keys())

def plot_history(history, figsize=(8,8)):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1, figsize=figsize)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    legend = plt.legend(frameon = 1)
    frame = legend.get_frame()
    frame.set_color('white')
    
    ## Accuracy
    plt.figure(2, figsize=figsize)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    legend = plt.legend(frameon = 1)
    frame = legend.get_frame()
    frame.set_color('white')
    plt.show()

plot_history(history_wmfl, figsize=(10,6))

plot_history(history_wmfl_2, figsize=(10,6))

# let's continue training and see if we can improve accuracy and loss

wide_model_fast_learn.load_weights('saved_models/weights.bestaugmented.from_scratch_wide_model_fast_learn_v2.hdf5')

history_wmfl_2 = wide_model_fast_learn.fit_generator(train_generator, validation_data=validation_generator,
                         steps_per_epoch=800, epochs=20, callbacks=[checkpointer, early_stopping])

wide_model_fast_learn.save('saved_models/wide_model_fast_learn.h5')



batch_size = 64

history_wmfl_3 = wide_model_fast_learn.fit_generator(train_generator, validation_data=validation_generator,
                         steps_per_epoch=150, epochs=10, callbacks=[checkpointer])

# appears this first model is going to top out in accuracy and bottom out in loss here... let's plot our progress so far

plot_history(history_wmfl_3, figsize=(12,6))

# trying new model with increase # of filters and "he_normal" kernel initializer.  "glorot_uniform" is default
# also adding dropout layer since it seems that our prior model experience overfitting 

new_model = Sequential()

new_model.add(BatchNormalization(input_shape=input_shape))
new_model.add(Conv2D(16, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model.add(MaxPooling2D(pool_size=(2, 2)))
new_model.add(BatchNormalization())


new_model.add(Conv2D(32, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model.add(MaxPooling2D(pool_size=(2, 2)))
new_model.add(BatchNormalization())

new_model.add(Conv2D(64, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model.add(MaxPooling2D(pool_size=(2, 2)))
new_model.add(BatchNormalization())

new_model.add(Conv2D(128, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model.add(MaxPooling2D(pool_size=(2, 2)))
new_model.add(Dropout(0.4))
new_model.add(BatchNormalization())

new_model.add(Conv2D(256, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model.add(MaxPooling2D(pool_size=(2, 2)))
new_model.add(Dropout(0.4))
new_model.add(BatchNormalization())

new_model.add(GlobalAveragePooling2D())

new_model.add(Dense(num_classes, activation='softmax'))

new_model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print(new_model.summary())

checkpointer = ModelCheckpoint(filepath='saved_models/weights.bestaugmented.from_scratch_saksham789.hdf5', 
                               verbose=1, save_best_only=True)

batch_size = 64

history = new_model.fit_generator(train_generator, validation_data=validation_generator,
                         steps_per_epoch=150, epochs=100, callbacks=[checkpointer])

plot_history(history,figsize=(12,6))

new_model.save('saved_models/new_model.h5')



# continue training?

# appears dropout might be too aggresive... 

new_model_2 = Sequential()

new_model_2.add(BatchNormalization(input_shape=input_shape))
new_model_2.add(Conv2D(16, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model_2.add(MaxPooling2D(pool_size=(2, 2)))
new_model_2.add(BatchNormalization())


new_model_2.add(Conv2D(32, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model_2.add(MaxPooling2D(pool_size=(2, 2)))
new_model_2.add(BatchNormalization())

new_model_2.add(Conv2D(64, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model_2.add(MaxPooling2D(pool_size=(2, 2)))
new_model_2.add(BatchNormalization())

new_model_2.add(Conv2D(128, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model_2.add(MaxPooling2D(pool_size=(2, 2)))
new_model_2.add(Dropout(0.2))
new_model_2.add(BatchNormalization())

new_model_2.add(Conv2D(256, (3, 3), strides=1, kernel_initializer='he_normal', activation='relu'))
new_model_2.add(MaxPooling2D(pool_size=(2, 2)))
new_model_2.add(Dropout(0.2))

new_model_2.add(GlobalAveragePooling2D())

new_model_2.add(Dense(num_classes, activation='softmax'))

new_model_2.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print(new_model_2.summary())

from keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir='C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\tbLogs\\CNN_from_scratch', 
                          histogram_freq=0, batch_size=64, write_graph=True, write_grads=False, write_images=False, 
                          embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)



batch_size = 64

checkpointer = ModelCheckpoint(filepath='saved_models/weights.bestaugmented.from_scratch_saksham789_v3.hdf5', 
                               verbose=1, save_best_only=True)

history = new_model_2.fit_generator(train_generator, validation_data=validation_generator,
                         steps_per_epoch=150, epochs=100, callbacks=[checkpointer, tensorboard])

new_model_2.save('saved_models/new_model_2_v2.h5')

plot_history(history, figsize=(12,6))

# load the  model

from keras.models import load_model

model = load_model('saved_models/wide_model_fast_learn.h5')
model.load_weights('saved_models/weights.bestaugmented.from_scratch_wide_model_fast_learn_v2.hdf5')





test_tensor=load_array('test_dataset.npy')

test_tensor = Normalize_Input(test_tensor)

test_labels =load_array('test_labels.npy')

datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, 
                             zoom_range=0.1, horizontal_flip=True)

test_generator = datagen.flow(x=test_tensor, y=test_labels, batch_size=batch_size)

# load the model
from keras.models import load_model

model = load_model('saved_models/new_model_2.h5')

accuracy = model.evaluate_generator(test_generator, max_queue_size=10)

print(accuracy)



