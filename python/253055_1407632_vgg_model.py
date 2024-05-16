import numpy as np
from keras.models import Sequential
from utils_channels_last import Vgg16BN
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K

import os

from tensorflow.python.client import device_lib
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
print(device_lib.list_local_devices())

# configure tensorflow before fitting model
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.Session(config=tf_config)

# changing directory for flow_from_directory method
os.chdir('C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets')

batch_size=12
num_classes = 120
image_size = 224
num_channels = 3

train_datagen = ImageDataGenerator(rotation_range=15, shear_range=0.1, channel_shift_range=20,
                                    width_shift_range=0.1,  height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True,
                                  validation_split=0.2)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory('cropped/train', target_size=(224,224),
            class_mode='categorical', shuffle=True, batch_size=batch_size, subset='training')

validation_generator = train_datagen.flow_from_directory('cropped/train', target_size=(224,224),
            class_mode='categorical', shuffle=True, batch_size=batch_size, subset='validation')

test_generator = test_datagen.flow_from_directory('cropped/test', target_size=(224,224),
            class_mode='categorical', shuffle=False, batch_size=batch_size)

base_model = Vgg16BN(include_top=False).model
x = base_model.output
x = Flatten()(x)
x = Dropout(0.4)(x)
# let's add two fully-connected layer
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
# and a final FC layer with 'softmax' activation since we are doing a multi-class problem 
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

from keras.callbacks import ModelCheckpoint

model.load_weights('saved_models/weights.vgg16_BN_finetuned.h5')

checkpointer = ModelCheckpoint(filepath='saved_models/weights.bestaugmented.pre_trained_vgg16_v3.hdf5', 
                               verbose=1, save_best_only=True)


history = model.fit_generator(train_generator, steps_per_epoch=800, epochs=25, 
                              validation_data=validation_generator,
                              callbacks=[checkpointer])

# saving what we have so far
model.save('saved_models/vgg16_BN_finetuned.h5')
model.save_weights('saved_models/weights.vgg16_BN_finetuned.h5')

# not sure if I need this later but saving the classes from the generator object and the respective indices
label_dict = train_generator.class_indices

# lets plot/visualize the model training progress

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('bmh')

font = {'family' : 'sans-serif',
        'weight' : 'medium',
        'size'   : 16}

plt.rc('font', **font)

def plot_history(history, figsize=(8,8)):
    '''
    Args: the history attribute of model.fit, figure size (defaults to (8,8))
    Description: Takes the history object and plots the loss and accuracy metrics (both train and validation)
    Returns: Plots of Loss and Accuracy from model training
    '''
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

plot_history(history, figsize=(10,6))

# looks like we're starting to overfit, let's see if we can improve by continuing to train with a slower learning rate 
# as gains in val_loss are becoming less frequent

model.optimizer.lr = 1e6
history = model.fit_generator(train_generator, steps_per_epoch=800, epochs=15, 
                        validation_data=validation_generator,
                       callbacks=[checkpointer])



plot_history(history, figsize=(10,6))



from sklearn.metrics import confusion_matrix, classification_report, accuracy_score 
import itertools

model.load_weights('saved_models/weights.vgg16_BN_finetuned.h5')

# now lets evaluate the model on our unseen test data

y_pred = model.predict_generator(test_generator, max_queue_size =10)

import pandas as pd

# create a dataframe of the predictions to find the most hallmark example of a class in the eyes of the model

results_df = pd.DataFrame(y_pred)
results_df.columns = list(label_dict.values())
results_df.head()

# find most pugly

image = results_df['pug'].idxmax()
image

# need index of all test images

folders = [x[0] for x in os.walk('test')][1:]
files = [os.listdir(f) for f in folders]

flattened_list = [y for x in files for y in x]



files = pd.DataFrame(flattened_list)
files.head()

# ok lets find the pug image

pug = files.iloc[image]
print(pug)

display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02110958-pug\n02110958_609.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)

# corgi
image = results_df['Pembroke'].idxmax()
image

corgi = files.iloc[image]
print(corgi)

display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02113023-Pembroke\n02113023_3913.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)

# let's look at the most misclassified pairs siberian husky and eskimo dog

image = results_df['Siberian_husky'].idxmax()

husky = files.iloc[image]
print(husky)

display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02110185-Siberian_husky\n02110185_13187.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)

#... now eskimo dog

image = results_df['Eskimo_dog'].idxmax()

eskimo = files.iloc[image]
print(eskimo)

display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02110185-Siberian_husky\n02110185_699.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)

#interestingly the model thinks that an image from the Siberian husky folder is the most "Eskimo Dog" even out of the eskimo dog images

# entlebucher

image = results_df['EntleBucher'].idxmax()

EntleBucher = files.iloc[image]
print(EntleBucher)

display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02108000-EntleBucher\n02108000_1462.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)

# Greater_Swiss_Mountain_dog
image = results_df['Greater_Swiss_Mountain_dog'].idxmax()

greater_swiss = files.iloc[image]
print(greater_swiss)

display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02107574-Greater_Swiss_Mountain_dog\n02107574_2665.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)

# Labrador_retriever, 2017 most popular breed

image = results_df['Labrador_retriever'].idxmax()

Labrador_retriever = files.iloc[image]
print(Labrador_retriever)

display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02099712-Labrador_retriever\n02099712_7866.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)

# Number 2, German_shepherd

image = results_df['German_shepherd'].idxmax()

German_shepherd = files.iloc[image]
print(German_shepherd)

display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02106662-German_shepherd\n02106662_16817.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)

# 3, golden_retriever

image = results_df['golden_retriever'].idxmax()

golden_retriever = files.iloc[image]
print(golden_retriever)

display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02099601-golden_retriever\n02099601_2994.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)

# 4, French_bulldog

image = results_df['French_bulldog'].idxmax()

French_bulldog = files.iloc[image]
print(French_bulldog)

display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02108915-French_bulldog\n02108915_3702.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)

# 6, beagle (#5 english bulldog is not in database)

image = results_df['beagle'].idxmax()

beagle = files.iloc[image]
print(beagle)

display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02088364-beagle\n02088364_959.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)

# malinois

image = results_df['malinois'].idxmax()

malinois = files.iloc[image]
print(malinois)

display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02105162-malinois\n02105162_6596.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)

# cairn

image = results_df['cairn'].idxmax()

cairn = files.iloc[image]
print(cairn)

display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02096177-cairn\n02096177_2842.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)

# vizsla

image = results_df['vizsla'].idxmax()

vizsla = files.iloc[image]
print(vizsla)

display_image = cv2.imread(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02100583-vizsla\n02100583_7522.jpg''')
display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(display_image)

y_pred = np.argmax(y_pred, axis=1).astype(int)
y_pred



test_labels = np.load(open('test_labels.npy','rb'))

cm = confusion_matrix(test_labels, y_pred)

#normalize the confusion matrix

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

breeds = list(label_dict.values())

import seaborn as sns
import pandas as pd

fig, ax = plt.subplots(figsize=(30, 26))
plt.title('Confusion Matrix on Test Images')
_ = sns.heatmap(cm, ax=ax, yticklabels=breeds, xticklabels=breeds, robust=True)

accuracy = model.evaluate_generator(test_generator, max_queue_size=10)

print(accuracy)

'''

# credit to: https://gist.github.com/nickynicolson/202fe765c99af49acb20ea9f77b6255e

def cm2df(cm, labels):
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata={}
        # columns
        for j, col_label in enumerate(labels): 
            rowdata[col_label]=cm[i,j]
        df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
    return df

df = cm2df(cm, breeds)
'''

df = pd.DataFrame(y_pred, columns=['predicted'], dtype='int')

df.head()

df['actual'] = pd.Series(test_labels).astype('int')
df['count'] = 1
df['count'] = df['count'].astype('int')

df.info()

# need to create dictionary to assign the numeric labels to string labels 
# first let's reverse the order of the label_dict (again this was datagen.class_indices)

label_dict = {y:x for x,y in label_dict.items()}

for key, value in label_dict.items():
    label_dict[key] = value[10:]

df.replace({"actual": label_dict}, inplace=True)
df.replace({"predicted": label_dict}, inplace=True)

df.head(25)

# let's take a look at the top 30 most confused pairs

misclass_df = df[df['actual'] != df['predicted']].groupby(['actual', 'predicted']).sum().sort_values(['count'], ascending=False).reset_index()
misclass_df['pair'] = misclass_df['actual'] + ' / ' + misclass_df['predicted']
misclass_df = misclass_df[['pair', 'count']].take(range(30))

misclass_df.sort_values(['count'], ascending=False).plot(kind='barh', figsize=(8, 10), x=misclass_df['pair'])
plt.title('Top 30 Misclassified Breed Pairs')

new_misclass_df = df[df['actual'] != df['predicted']].groupby(['actual', 'predicted']).sum().sort_values(['count'], ascending=True).reset_index()
new_misclass_df['pair'] = new_misclass_df['actual'] + ' / ' + new_misclass_df['predicted']

new_misclass_df.tail()


new_misclass_df.tail(30).plot(kind='barh', figsize=(8, 10), x='pair', y='count', color='red')
plt.title('Top 30 Misclassified Breed Pairs')

# https://stackoverflow.com/questions/41695844/keras-showing-images-from-data-generator?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
'''
x,y = train_generator.next()
for i in range(0,5):
    image = x[i]
    plt.imshow(image.transpose(2,1,0))
    plt.show()
'''

# finally let's get the classification report in case we need it

print(classification_report(y_true=test_labels, y_pred=y_pred, target_names=list(label_dict.values())))

# now let's visualize whats going on in the network itself as we pass images through
# adopted from Keras documentation: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

layer_dict = dict([(layer.name, layer) for layer in model.layers])

from keras import backend as K

from __future__ import print_function

from scipy.misc import imsave
import time

img_width = 224
img_height = 224

layer_name = 'conv2d_5'

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

input_img = model.input

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

get_ipython().run_cell_magic('capture', '', "\nkept_filters = []\nfor filter_index in range(200):\n\n    # we only scan through the first 200 filters,\n    # but there are actually 512 of them\n    print('Processing filter %d' % filter_index)\n    start_time = time.time()\n\n    # we build a loss function that maximizes the activation\n    # of the nth filter of the layer considered\n    layer_output = layer_dict[layer_name].output\n    if K.image_data_format() == 'channels_first':\n        loss = K.mean(layer_output[:, filter_index, :, :])\n    else:\n        loss = K.mean(layer_output[:, :, :, filter_index])\n\n    # we compute the gradient of the input picture wrt this loss\n    grads = K.gradients(loss, input_img)[0]\n\n    # normalization trick: we normalize the gradient\n    grads = normalize(grads)\n\n    # this function returns the loss and grads given the input picture\n    iterate = K.function([input_img], [loss, grads])\n\n    # step size for gradient ascent\n    step = 1.\n\n    # we start from a gray image with some random noise\n    if K.image_data_format() == 'channels_first':\n        input_img_data = np.random.random((1, 3, img_width, img_height))\n    else:\n        input_img_data = np.random.random((1, img_width, img_height, 3))\n    input_img_data = (input_img_data - 0.5) * 20 + 128\n\n    # we run gradient ascent for 20 steps\n    for i in range(20):\n        loss_value, grads_value = iterate([input_img_data])\n        input_img_data += grads_value * step\n\n        print('Current loss value:', loss_value)\n        if loss_value <= 0.:\n            # some filters get stuck to 0, we can skip them\n            break\n\n    # decode the resulting input image\n    if loss_value > 0:\n        img = deprocess_image(input_img_data[0])\n        kept_filters.append((img, loss_value))\n    end_time = time.time()\n    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))")

# we will stich the best 64 filters on a 8 x 8 grid.
n = 8

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))


# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

# save the result to disk
imsave('stitched_filters_%dx%d_2.png' % (n, n), stitched_filters)

# end keras tutorial

# let's visualize how the model interprets the image data. let's start using a picture that we're familiar with, Pepsi.
# adopted from https://github.com/erikreppel/visualizing_cnns/blob/master/visualize_cnns.ipynb

import cv2

pepsi = cv2.imread(r'''C:\Users\Garrick\Downloads\pepsi.jpg''')
pepsi = cv2.cvtColor(pepsi, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
plt.imshow(pepsi)

pepsi.shape

pepsi = cv2.resize(pepsi,(224,224))

pepsi = np.reshape(pepsi,[1,224,224,3])

pepsi.shape

pepsi = np.rollaxis(pepsi, 3, 1)

pepsi.shape

# image is now in shape appropriate for VGG16BN

pepsi_predict_proba = model.predict(pepsi)

pepsi_predict_class = np.argmax(pepsi_predict_proba)
label_dict[pepsi_predict_class]

# bummer not a good prediction, let's see how confident the model was

pepsi_top_prob = np.max(pepsi_predict_proba)
pepsi_top_prob

#let's create a function to apply these same steps over other images

def predict_image(filepath, model=model):
    '''
    Takes a single image and returns class/breed prediction and accuracy
    Dependencies: needs os, cv2 and keras, dependency file or perhaps annex to docker
    Args: filepath of image (no quotes), optional: specify the model, will default to current model in environment
    '''
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #cv2 uses BGR, so reverting back to RGB
    print('Uploaded Image...')
    plt.imshow(img)
    
    #resize and reshape for model input
    img = cv2.resize(img,(224,224))
    img = np.reshape(img,[1,224,224,3])
    img = np.rollaxis(img, 3, 1)
    
    #predict
    prediction = model.predict(img)
    
    #print results
    class_predict = np.argmax(prediction)
    breed = label_dict[class_predict]
    print('Woof! The model predicted the breed as...{}!'.format(breed))
    top_prob = np.max(prediction)
    print('...with a confidence of {0:.2f}%.'.format(top_prob*100))
    
          

predict_image('C:\\Users\\Garrick\\Downloads\\IMG_20180221_073001.jpg')

predict_image('C:\\Users\\Garrick\\Downloads\\pug.jpg')

#yaaaaaaaaas
# this is especially encouraging as pug/bull mastiff was the 2nd most common misclassification

predict_image('C:\\Users\\Garrick\\Downloads\\maggie.jpg')

# trick question, even my parents don't know what kind of dog Maggie is

predict_image('C:\\Users\\Garrick\\Downloads\\marnie.jpg')

predict_image('C:\\Users\\Garrick\\Downloads\\barkley.jpg')

# ok so which breed do I look like?

predict_image('C:\\Users\\Garrick\\Downloads\\1510749_671062136947_5867944268483557830_n.jpg')

# in conjunction with the human-performance study, passing in the images for the model
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02086910-papillon\n02086910_1775.jpg''')

# 2 of 10
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02109961-Eskimo_dog\n02109961_18381.jpg''')

# 3 of 10
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02108000-EntleBucher\n02108000_1462.jpg''')

# 4 of 10
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02096177-cairn\n02096177_1390.jpg''')

# 5 of 10
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02110063-malamute\n02110063_566.jpg''')

# 6 of 10
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02101006-Gordon_setter\n02101006_3379.jpg''')

# 7 of 10
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02106166-Border_collie\n02106166_152.jpg''')

# 8 of 10
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02086240-Shih-Tzu\n02086240_3424.jpg''')

# 10 of 10 (the 9th was Sir Charles Barkely)
predict_image(r'''C:\Users\Garrick\Documents\Springboard\Capstone Project 2\datasets\test\n02108551-Tibetan_mastiff\n02108551_1543.jpg''')



