from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense
import numpy as np
import math

# fixed random seed to have consistent results
np.random.seed(123)

img_width, img_height = 300, 300

train_data_dir = 'data/train'
validation_data_dir = 'data/test'
nb_train_samples = 3000
nb_validation_samples = 300
epochs = 5
batch_size = 600

input_shape = (img_height, img_width, 3)

model = Sequential()
model.add(Dense(3, input_shape=input_shape))
model.add(Flatten())
model.add(Dense(32, activation = 'relu')) # 3 hidden layers now
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adadelta',  metrics=['accuracy'])

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
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch= math.ceil(nb_train_samples / batch_size),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps= math.ceil(nb_validation_samples/batch_size))



