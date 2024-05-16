# for DSX, need to switch to the right directory. Detect using path name.
s = get_ipython().magic('pwd')
if s.startswith('/gpfs'):
    get_ipython().magic('cd ~/deep-learning-workshop/')
    

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import os
import glob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

# for making plots prettier
import seaborn as sns 
sns.set_style('white')

from __future__ import print_function
np.random.seed(1331)  # for reproducibility

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers

from display import visualize_keras_model, plot_training_curves

data_root = os.path.expanduser("~/data/cats_dogs")
train_data_dir = os.path.join(data_root, 'train')
validation_data_dir = os.path.join(data_root, 'validation')

# Make sure we have the expected numbers of images
for d in [train_data_dir, validation_data_dir]:
    for category in ['cats', 'dogs']:
        print("{}/{}: {}".format(
                d, category, len(os.listdir(os.path.join(d, category)))))

train_cats = os.path.join(train_data_dir,'cats')
train_dogs = os.path.join(train_data_dir,'dogs')

def viz_dir(dirpath):
    image_paths = os.listdir(dirpath)

    fig, axs = plt.subplots(2,5, figsize=(11,2.5))
    for i, img_path in enumerate(np.random.choice(image_paths, 10, replace=False)):
        img = Image.open(os.path.join(dirpath, img_path))
        ax = axs[i//5, i%5]
        ax.imshow(img)
        ax.axis('off')
        
viz_dir(train_cats)
viz_dir(train_dogs)

# dimensions of our images.
img_width, img_height = 150, 150

nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

visualize_keras_model(model)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')

plot_training_curves(history.history);

from keras.preprocessing.image import ImageDataGenerator
from keras import applications

top_model_weights_path = 'bottleneck_fc_model.h5'
nb_train_samples = 2000
nb_validation_samples = 800

epochs = 50
batch_size = 16

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    # Note: train_data.shape = (2000, 4, 4, 512)
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    return history

save_bottlebeck_features()

history = train_top_model()

plot_training_curves(history.history);

# Reset things...
from keras.layers.core import K
K.clear_session() 

# path to the model weights files.
weights_path = os.path.expanduser('~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

epochs = 50
batch_size = 16

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', 
                           include_top=False,
 # Add below line to gist to fix error:
 # ValueError: The shape of the input to "Flatten" is not fully
 # defined (got (None, None, 512). We can use our own
 # width and height because we're only keeping the convolutional layers
                           input_shape=(img_height,img_width,3))
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# Now we have the VGG model with our own layer on top
visualize_keras_model(model)

# Easier to count layers in this form. We want to freeze the first 
# 4 conv->pool blocks, which works out to be the first 15 layers
# (note that Keras blog post says 25--seems like a typo!)
model.layers

# set the first 15 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# fine-tune the model
epochs = 20
history = model.fit_generator(
    train_generator, 
    steps_per_epoch=nb_train_samples/batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples/batch_size)

plot_training_curves(history.history);

full_model_weights_path = 'full_model_weights.h5'
model.save_weights(full_model_weights_path)

