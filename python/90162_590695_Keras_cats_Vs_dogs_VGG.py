import os
import numpy as np
from keras import applications, optimizers
from keras.layers import Input,Dense, Dropout, Flatten
from keras.models import Sequential,Model
from keras.preprocessing.image import ImageDataGenerator

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = os.path.join(os.getcwd(), 'fc_model.h5')
train_data_dir = os.path.join(os.getcwd(), 'data', 'cats_and_dogs_small', 'train')
validation_data_dir = os.path.join(os.getcwd(), 'data', 'cats_and_dogs_small', 'validation')
img_width, img_height = 150, 150
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 4 #more than enough to get a good result
batch_size = 16

datagen = ImageDataGenerator(rescale=1. / 255)

# build the VGG16 network
print('Reading vgg')
model = applications.VGG16(include_top=False, weights='imagenet',input_shape=(img_width, img_height, 3))
model.summary()

generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size)

np.save('bottleneck_features_train.npy', bottleneck_features_train)
print('\nSaved bottleneck_features_train\n')

generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

bottleneck_features_validation = model.predict_generator(
    generator, nb_validation_samples // batch_size)

np.save('bottleneck_features_validation.npy',  bottleneck_features_validation)
print('\n--Saved bottleneck_features_validation--')

train_data = np.load('bottleneck_features_train.npy')

train_labels = np.array([0] * int(nb_train_samples/2 ) + [1] * int(nb_train_samples/2 ))

validation_data = np.load('bottleneck_features_validation.npy')

validation_labels = np.array([0] * int(nb_validation_samples/2 ) + [1] * int(nb_validation_samples/2))

print(train_data.shape)
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(validation_data, validation_labels))

model.save_weights(top_model_weights_path)

#Using generated model with layers of vgg

input_tensor = Input(shape=(150,150,3))
base_model = applications.VGG16(weights='imagenet',include_top= False,input_tensor=input_tensor)
print('VGG model')
base_model.summary()

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
top_model.load_weights('fc_model.h5') #optional - to load the already saved weights
model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

# set the first 15 layers (up to the conv block 4) to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer and a very slow learning rate.
model.compile(loss='binary_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

print('\nAugmenting train data')
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')


print('\nScaling test data')
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')


# fine-tune the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('VGG_cats_Vs_dogs.h5')

