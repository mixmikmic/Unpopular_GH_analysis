import os
import sys
import pandas as pd
import numpy as np
import PIL

seed = 16
np.random.seed(seed)

from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

#check using system GPU for processing

from tensorflow.python.client import device_lib
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
print(device_lib.list_local_devices())

# copied over the train, validate and test sets for 5 randomly selected breeds

os.chdir('C:\\Users\\Garrick\Documents\\Springboard\\Capstone Project 2\\datasets_subset1')

train_datagen = ImageDataGenerator(rotation_range=15, shear_range=0.1, channel_shift_range=20,
                                    width_shift_range=0.1,  height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True,
                                    fill_mode='nearest', rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 25

train_generator = train_datagen.flow_from_directory('subset_train', target_size=(224,224), color_mode='rgb',
            class_mode='categorical', shuffle=False, batch_size=batch_size)

validation_generator = validation_datagen.flow_from_directory('subset_val', target_size=(224,224), color_mode='rgb',
            class_mode='categorical', shuffle=False, batch_size=batch_size)


test_generator = test_datagen.flow_from_directory('subset_test', target_size=(224,224), color_mode='rgb',
            class_mode='categorical', shuffle=False, batch_size=batch_size)

# reminder to self... flow_from_directory infers the class labels

# importing keras modules and setting up a few parameters, instantiating early stopping

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import keras.utils
from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

import tensorflow as tf
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.99
# tf_config.gpu_options.allow_growth = True **this causes python to crash, error: las.cc:444] failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED
sess = tf.Session(config=tf_config)

input_shape = (224,224, 3)
num_classes = 5

# will create a few different models.... initial base model 

base_model = Sequential()
base_model.add(Conv2D(64, (11, 11), strides=4, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
base_model.add(MaxPooling2D(pool_size=(2, 2)))

base_model.add(Conv2D(64, (4, 4), strides=2, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
base_model.add(MaxPooling2D(pool_size=(2, 2)))

base_model.add(Conv2D(64, (4, 4), strides=2, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
base_model.add(Flatten())

base_model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
base_model.add(Dropout(0.2))
base_model.add(Dense(num_classes, activation='softmax'))
    
# Compile model
epochs = 10
lrate = 0.003
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
base_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(base_model.summary())

# train base_model

base_model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=100, epochs=epochs, callbacks=[early_stopping])

# same model, more epochs (10 -> 50) and fewere steps per epoch, prior model params saw train and validate accuracies double, expecting the model to be within 80% acc

base_model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=50, epochs=25, callbacks=[early_stopping])

# taking the base model and adding more hidden layers

deep_model = Sequential()

deep_model = Sequential()
deep_model.add(Conv2D(64, (11, 11), strides=4, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
deep_model.add(MaxPooling2D(pool_size=(2, 2)))

deep_model.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model.add(MaxPooling2D(pool_size=(2, 2)))

deep_model.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model.add(MaxPooling2D(pool_size=(2, 2)))

deep_model.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model.add(Flatten())

deep_model.add(Dense(288, activation='relu', kernel_constraint=maxnorm(3)))
deep_model.add(Dropout(0.2))
deep_model.add(Dense(num_classes, activation='softmax'))
    
# Compile model
epochs = 10
lrate = 0.003
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
deep_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(deep_model.summary())

# train deeper model

deep_model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=100, epochs=epochs, callbacks=[early_stopping])

# looks like the deeper model overfits the training data, but performs better on the validation data... let's train for more epochs

deep_model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=100, epochs=50, callbacks=[early_stopping])

# deeper model ran into early stopping on the validation set

# lets try base model with more epochs

base_model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=100, epochs=25, callbacks=[early_stopping])

# deep model with Adam optimizer

# taking the base model and adding more hidden layers



deep_model_Adam = Sequential()

deep_model_Adam = Sequential()
deep_model_Adam.add(Conv2D(64, (11, 11), strides=4, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
deep_model_Adam.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_Adam.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_Adam.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_Adam.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_Adam.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_Adam.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_Adam.add(Flatten())

deep_model_Adam.add(Dense(288, activation='relu', kernel_constraint=maxnorm(3)))
deep_model_Adam.add(Dropout(0.2))
deep_model_Adam.add(Dense(num_classes, activation='softmax'))
    
# Compile model
adam_op = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
deep_model_Adam.compile(loss='categorical_crossentropy', optimizer=adam_op, metrics=['accuracy'])
print(deep_model_Adam.summary())

deep_model_Adam.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=100, epochs=25, callbacks=[early_stopping])

# tweaked deep model w/ Adam optimizer.  Deeper network topology near the input (less convolution than prior models), more FC nodes

deep_model_Adam_2 = Sequential()

deep_model_Adam_2 = Sequential()
deep_model_Adam_2.add(Conv2D(64, (8, 8), strides=2, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
deep_model_Adam_2.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_Adam_2.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_Adam_2.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_Adam_2.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_Adam_2.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_Adam_2.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_Adam_2.add(Flatten())

deep_model_Adam_2.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
deep_model_Adam_2.add(Dropout(0.2))
deep_model_Adam_2.add(Dense(num_classes, activation='softmax'))
    
# Compile model
adam_op = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
deep_model_Adam_2.compile(loss='categorical_crossentropy', optimizer=adam_op, metrics=['accuracy'])
print(deep_model_Adam_2.summary())

deep_model_Adam_2.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=50, epochs=50, callbacks=[early_stopping])

# tweaked deep model w/ RMSProp optimizer again with Deeper network topology near the input (less convolution than prior models), more FC nodes

deep_model_RMS = Sequential()

deep_model_RMS = Sequential()
deep_model_RMS.add(Conv2D(64, (8, 8), strides=2, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
deep_model_RMS.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_RMS.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_RMS.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_RMS.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_RMS.add(MaxPooling2D(pool_size=(2, 2)))

deep_model_RMS.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
deep_model_RMS.add(Flatten())

deep_model_RMS.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
deep_model_RMS.add(Dropout(0.2))
deep_model_RMS.add(Dense(num_classes, activation='softmax'))
    
# Compile model
deep_model_RMS.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(deep_model_RMS.summary())

deep_model_RMS.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=100, epochs=50, callbacks=[early_stopping])

# so more layers doesn't work.... let us keep the standard 3 CONV layers and widen the toplogy

wide_model = Sequential()
wide_model.add(Conv2D(32, (3, 3), strides=1, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
wide_model.add(MaxPooling2D(pool_size=(2, 2)))

wide_model.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model.add(MaxPooling2D(pool_size=(2, 2)))

wide_model.add(Conv2D(32, (3, 3), strides=2, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model.add(MaxPooling2D(pool_size=(2, 2)))
wide_model.add(Flatten())

wide_model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
wide_model.add(Dropout(0.2))
wide_model.add(Dense(num_classes, activation='softmax'))
    
# Compile model
epochs = 10
wide_model.compile(loss='categorical_crossentropy', optimizer=adam_op, metrics=['accuracy'])
print(wide_model.summary())

wide_model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=50, epochs=50, callbacks=[early_stopping])

# wider doesn't necessarily work... however, slowing the learning rate seems to having a positive impact. same model as above, decrease LR

wide_model_slow_learn = Sequential()
wide_model_slow_learn.add(Conv2D(32, (3, 3), strides=1, input_shape=input_shape, padding='valid', activation='relu', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))

wide_model_slow_learn.add(Conv2D(32, (3, 3), strides=1, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))

wide_model_slow_learn.add(Conv2D(32, (3, 3), strides=2, activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(MaxPooling2D(pool_size=(2, 2)))
wide_model_slow_learn.add(Flatten())

wide_model_slow_learn.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
wide_model_slow_learn.add(Dropout(0.2))
wide_model_slow_learn.add(Dense(num_classes, activation='softmax'))
    
# Compile model

adam_op = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
wide_model_slow_learn.compile(loss='categorical_crossentropy', optimizer=adam_op, metrics=['accuracy'])
print(wide_model_slow_learn.summary())

wide_model_slow_learn.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=50, epochs=50, callbacks=[early_stopping])

# it appears a slower learning rate might be key in allowing prior models to train for more epochs... 
# let's try a few earlier models with a decreased learning rate

adam_op = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
deep_model_Adam.compile(loss='categorical_crossentropy', optimizer=adam_op, metrics=['accuracy'])
print(deep_model_Adam.summary())

deep_model_Adam.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=50, epochs=50, callbacks=[early_stopping])

# let's try tye base model w/ decreased learning rate and Adam optimizer (vs. SGD)

base_model.compile(loss='categorical_crossentropy', optimizer=adam_op, metrics=['accuracy'])
print(base_model.summary())

base_model.fit_generator(train_generator, validation_data=validation_generator,
                    steps_per_epoch=50, epochs=50, callbacks=[early_stopping])

# let's test on these iterations of the base, deep and wide models

base_scores = base_model.evaluate_generator(test_generator, steps=25)
print("Accuracy: %.2f%%" % (base_scores[1]*100))

deep_model_Adam_scores = deep_model_Adam.evaluate_generator(test_generator, steps=25)
print("Accuracy: %.2f%%" % (deep_model_Adam_scores[1]*100))

wide_model_slow_learn_scores = wide_model_slow_learn.evaluate_generator(test_generator, steps=25)
print("Accuracy: %.2f%%" % (wide_model_slow_learn_scores[1]*100))

# saving models and weights just in case... will need to retrain on broader image sets anyways
base_model.save('subset_base_model.h5')
deep_model_Adam.save('subset_deep_model_Adam.h5')
wide_model_slow_learn.save('subset_wide_model_slow_learn.h5')





