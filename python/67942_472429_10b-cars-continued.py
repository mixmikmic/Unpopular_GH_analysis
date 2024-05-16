get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# system
import os
import glob
import itertools as it
import operator
from collections import defaultdict
from StringIO import StringIO

# other libraries
import cPickle as pickle
import numpy as np 
import pandas as pd
import scipy.io  # for loading .mat files
import scipy.misc # for imresize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import seaborn as sns
import requests

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, GlobalAveragePooling2D
from keras.utils import np_utils

# https://github.com/fchollet/keras/issues/4499
from keras.layers.core import K
from keras.callbacks import TensorBoard

# for name scopes to make TensorBoard look prettier (doesn't work well yet)
import tensorflow as tf 

# my code
from display import (visualize_keras_model, plot_training_curves,
                     plot_confusion_matrix)
from helpers import combine_histories

get_ipython().magic('matplotlib inline')
sns.set_style("white")
p = sns.color_palette()

# repeatability:
np.random.seed(42)

data_root = os.path.expanduser("~/data/cars")

from collections import namedtuple
Example = namedtuple('Example',
                     ['rel_path', 'x1', 'y1', 'x2','y2','cls','test'])

# Load data we saved in 10-cars.ipynb
with open('class_details.pkl') as f:
    loaded = pickle.load(f)
    macro_classes = loaded['macro_classes']
    macro_class_map = loaded['macro_class_map']
    cls_tuples = loaded['cls_tuples']
    classes = loaded['classes']
    examples = loaded['examples']
    by_class = loaded['by_class']
    by_car_type = loaded['by_car_type']

macro_class_map

resized_path = os.path.join(data_root,'resized_car_ims') 

def gray_to_rgb(im):
    """
    Noticed (due to array projection error in code below) that there is at least
    one grayscale image in the dataset.
    We'll use this to convert.
    """
    w, h = im.shape
    ret = np.empty((w,h,3), dtype=np.uint8)
    ret[:,:,0] = im
    ret[:,:,1] = im
    ret[:,:,2] = im
    return ret

def load_examples(by_class, cls, limit=None):
    """
    Load examples for a class. Ignores test/train distinction -- 
    we'll do our own train/validation/test split later.
    
    Args:
        by_class: our above dict -- class_id -> [Example()]
        cls: which class to load
        limit: if not None, only load this many images.
        
    Returns:
        list of (X,y) tuples, one for each image.
            X: 3x227x227 ndarray of type uint8
            Y: class_id (will be equal to cls)
    """
    res = []
    to_load = by_class[cls]
    if limit:
        to_load = to_load[:limit]

    for ex in to_load:
        # load the resized image!
        img_path = os.path.join(data_root, 
                        ex.rel_path.replace('car_ims', 'resized_car_ims'))
        img = mpimg.imread(img_path)
        # handle any grayscale images
        if len(img.shape) == 2:
            img = gray_to_rgb(img)
        res.append((img, cls))
    return res

def split_examples(xs, valid_frac, test_frac):
    """
    Randomly splits the xs array into train, valid, test, with specified 
    percentages. Rounds down.
    
    Returns:
        (train, valid, test)
    """
    assert valid_frac + test_frac < 1
    
    n = len(xs)
    valid = int(valid_frac * n)
    test = int(test_frac * n)
    train = n - valid - test
    
    # don't change passed-in list
    shuffled = xs[:]
    np.random.shuffle(shuffled)

    return (shuffled[:train], 
            shuffled[train:train + valid], 
            shuffled[train + valid:])

# quick test
split_examples(range(10), 0.2, 0.4)

# Look at training data -- there's so little we can look at all of it

def plot_data(xs, ys, predicts):
    """Plot the images in xs, with corresponding correct labels
    and predictions.
    
    Args:
        xs: RGB or grayscale images with float32 values in [0,1].
        ys: one-hot encoded labels
        predicts: probability vectors (same dim as ys, normalized e.g. via softmax)
    """
    
    # sort all 3 by ys
    xs, ys, ps = zip(*sorted(zip(xs, ys, predicts), 
                             key=lambda tpl: tpl[1][0]))
    n = len(xs)
    rows = (n+9)/10
    fig, plots = plt.subplots(rows,10, sharex='all', sharey='all',
                             figsize=(20,2*rows), squeeze=False)
    for i in range(n):
        # read the image
        ax = plots[i // 10, i % 10]
        ax.axis('off')
        img = xs[i].reshape(227,227,-1) 

        if img.shape[-1] == 1: # Grayscale
            # Get rid of the unneeded dimension
            img = img.squeeze()
            # flip grayscale:
            img = 1-img 
            
        ax.imshow(img)
        # dot with one-hot vector picks out right element
        pcorrect = np.dot(ps[i], ys[i]) 
        if pcorrect > 0.8:
            color = "blue"
        else:
            color = "red"
        ax.set_title("{}   p={:.2f}".format(int(ys[i][0]), pcorrect),
                     loc='center', fontsize=18, color=color)
    return fig

# normalize the data, this time leaving it in color
def normalize_for_cnn(xs):
    ret = (xs / 255.0)
    return ret

def image_from_url(url):
    response = requests.get(url)
    img = Image.open(StringIO(response.content))
    return img

# Load images
IMG_PER_CAR = None # 20 # None to use all
valid_frac = 0.2
test_frac = 0.2

train = []
valid = []
test = []
for car_type, model_tuples in by_car_type.items():
    macro_class_id = macro_class_map[car_type]
    
    for model_tpl in model_tuples:
        cls = model_tpl[0]
        examples = load_examples(by_class, cls, limit=IMG_PER_CAR)
        # replace class labels with the id of the macro class
        examples = [(X, macro_class_id) for (X,y) in examples]
        # split each class separately, so all have same fractions of 
        # train/valid/test
        (cls_train, cls_valid, cls_test) = split_examples(
            examples,
            valid_frac, test_frac)
        # and add them to the overall train/valid/test sets
        train.extend(cls_train)
        valid.extend(cls_valid)
        test.extend(cls_test)

# ...and shuffle to make training work better.
np.random.shuffle(train)
np.random.shuffle(valid)
np.random.shuffle(test)

# We have lists of (X,Y) tuples. Let's unzip into lists of Xs and Ys.
X_train, Y_train = zip(*train)
X_valid, Y_valid = zip(*valid)
X_test, Y_test = zip(*test)

# and turn into np arrays of the right dimension.
def convert_X(xs):
    '''
    Take list of (w,h,3) images.
    Turn into an np array, change type to float32.
    '''
    return np.array(xs).astype('float32')
    
X_train = convert_X(X_train)
X_valid = convert_X(X_valid)
X_test = convert_X(X_test)

X_train.shape

def convert_Y(ys, macro_classes):
    '''
    Convert to np array, make one-hot.
    Already ensured they're sequential from zero.
    '''
    n_classes = len(macro_classes)
    return np_utils.to_categorical(ys, n_classes)

Y_train = convert_Y(Y_train, macro_classes)
Y_valid = convert_Y(Y_valid, macro_classes)
Y_test = convert_Y(Y_test, macro_classes)

Y_train.shape

# normalize the data, this time leaving it in color
X_train_norm = normalize_for_cnn(X_train)
X_valid_norm = normalize_for_cnn(X_valid)
X_test_norm = normalize_for_cnn(X_test)

# Let's use more or less the same model to start (num classes changes)
def cnn_model2(use_dropout=True):
    model = Sequential()
    nb_filters = 16
    pool_size = (2,2)
    filter_size = 3
    nb_classes = len(macro_classes)
    
    with tf.name_scope("conv1") as scope:
        model.add(Convolution2D(nb_filters, filter_size, 
                            input_shape=(227, 227, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        if use_dropout:
            model.add(Dropout(0.5))

    with tf.name_scope("conv2") as scope:
        model.add(Convolution2D(nb_filters, filter_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        if use_dropout:
            model.add(Dropout(0.5))

    with tf.name_scope("conv3") as scope:
        model.add(Convolution2D(nb_filters, filter_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        if use_dropout:
            model.add(Dropout(0.5))

    with tf.name_scope("dense1") as scope:
        model.add(Flatten())
        model.add(Dense(16))
        model.add(Activation('relu'))
        if use_dropout:
            model.add(Dropout(0.5))

    with tf.name_scope("softmax") as scope:
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
    return model

# Uncomment if getting a "Invalid argument: You must feed a value
# for placeholder tensor ..." when rerunning training. 
# K.clear_session() # https://github.com/fchollet/keras/issues/4499
    

model3 = cnn_model2()
model3.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# This model will train slowly, so let's checkpoint it periodically
from keras.callbacks import ModelCheckpoint

recompute = False

if recompute:
#     # Save info during computation so we can see what's happening
#     tbCallback = TensorBoard(
#         log_dir='./graph', histogram_freq=1, 
#         write_graph=False, write_images=False)

    checkpoint = ModelCheckpoint('macro_class_cnn_checkpoint.5',
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True, mode='max',
                                 save_weights_only=True)

    # Fit the model! Using a bigger batch size and fewer epochs
    # because we have ~10K training images now instead of 100.
    history = model3.fit(
        X_train_norm, Y_train,
        batch_size=64, nb_epoch=50, verbose=1,
        validation_data=(X_valid_norm, Y_valid),
        callbacks=[checkpoint]
    )
else:
    model3.load_weights('macro_class_cnn.5')

# change to True to save
if False:
    model3.save('macro_class_cnn.h5')

# Get the predictions
predict_train = model3.predict(X_train_norm)
predict_valid = model3.predict(X_valid_norm)
predict_test = model3.predict(X_test_norm)

plot_confusion_matrix(Y_test, predict_test, macro_classes,
                      normalize=False,
                      title="Test confusion matrix");

plot_confusion_matrix(Y_train, predict_train, macro_classes,
                      title="Train confusion matrix");

# Normalized to see per-class behavior better
plot_confusion_matrix(Y_train, predict_train, macro_classes,                      
                      title="Train confusion matrix", normalize=True);

# What's our class balance
xs, counts = np.unique(np.argmax(Y_train, axis=1),return_counts=True)
plt.bar(xs, counts, tick_label=macro_classes, align='center')

predict_train_labels = np.argmax(predict_train, axis=1)
correct_labels = np.argmax(Y_train, axis=1)

correct_train = np.where(predict_train_labels==correct_labels)[0]
wrong_train = np.where(predict_train_labels!=correct_labels)[0]
percent = 100 * len(correct_train)/float(len(correct_labels))
print("Training: {:.2f}% correct".format(percent))

n_to_view = 20
subset = np.random.choice(correct_train, n_to_view, replace=False)
fig = plot_data(X_train_norm[subset], Y_train[subset], predict_train[subset]);
fig.suptitle("Correct predictions")

n_to_view = 20
subset = np.random.choice(wrong_train, n_to_view, replace=False)
fig = plot_data(X_train_norm[subset], Y_train[subset], predict_train[subset]);
fig.suptitle("Wrong predictions")

correct_coupe = np.where((predict_train_labels==correct_labels) & (correct_labels==macro_class_map['Coupe']))[0]
wrong_coupe = np.where((predict_train_labels!=correct_labels) & (correct_labels==macro_class_map['Coupe']))[0]

n_to_view = 40
subset = np.random.choice(correct_coupe, n_to_view, replace=False)
fig = plot_data(X_train_norm[subset], Y_train[subset], predict_train[subset]);
fig.suptitle("Correct coupe predictions")

subset = np.random.choice(wrong_coupe, n_to_view, replace=False)
fig = plot_data(X_train_norm[subset], Y_train[subset], predict_train[subset]);
fig.suptitle("Wrong coupe predictions", fontsize=18)

get_ipython().system('pip install keras_squeezenet')

from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

model = SqueezeNet()

# Oops -- screwed up the first training image (see note about preprocess_input below)
X_train[0]

img = X_train[1]
plt.imshow(img/255.0)
x = img# image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# preprocess_input modifies its argument!
x = preprocess_input(x.copy())

preds = model.predict(x)

print('Predicted:', decode_predicktions(preds))

# screwed up X_train[0] earlier. Rather than rerun, I tried to hack/fix it manually 
# (didn't really work, but it's just one image, so decided not to care). Now it looks funny, and is a good
# reminder to check your data throughout your pipeline, not just once at the beginning...
plt.imshow(X_train[0])

model.summary()

# We want to pull out the activations before conv10...

from keras.models import Model

# Get input
new_input = model.input
# Find the layer to connect
hidden_layer = model.get_layer('drop9').output
# Build a new model
bottleneck_model = Model(new_input, hidden_layer)
bottleneck_model.summary()

16000 * 13 * 13 * 512 * 4 / 2**20

train_subset = 2000
valid_subset = 1000

def save_bottlebeck_features(bottleneck_model, xs, name):
    # don't change the param!
    xs = preprocess_input(xs.copy())
    bottleneck_features = bottleneck_model.predict(xs)
    
    with open('cars_bottleneck_features_{}.npy'.format(name), 'w') as f:
        np.save(f, bottleneck_features)

def save_labels(ys, name):
    with open('cars_bottleneck_labels_{}.npy'.format(name), 'w') as f:
        np.save(f, ys)

        
if False: # change to True to recompute  
    save_bottlebeck_features(bottleneck_model, X_train[:train_subset], 'train_subset')
    save_bottlebeck_features(bottleneck_model, X_valid[:valid_subset], 'valid_subset')
    # save_bottlebeck_features(bottleneck_model, X_test, 'test')
    
    save_labels(Y_train[:train_subset], 'train_subset')
    save_labels(Y_valid[:valid_subset], 'valid_subset')

get_ipython().system('ls -lh cars*')

def load_features(name):
    with open('cars_bottleneck_features_{}.npy'.format(name), 'r') as f:
        return np.load(f)

def load_labels(name):
    with open('cars_bottleneck_labels_{}.npy'.format(name)) as f:
        return np.load(f)


top_model_weights_path = 'cars_bottleneck_fc_model.h5'    
    
# Now let's train the model -- we'll put the same squeezenet structure, just with fewer classes
def make_top_model():
    inputs = Input((13,13,512))
    x = Convolution2D(len(macro_classes), (1, 1), padding='valid', name='new_conv10')(inputs)
    x = Activation('relu', name='new_relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)

    model = Model(inputs, out, name='squeezed_top')
    
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

top_model = make_top_model()
print(top_model.summary())

train_data = load_features('train_subset')
train_labels = load_labels('train_subset')

valid_data = load_features('valid_subset')
valid_labels = load_labels('valid_subset')

epochs = 50
batch_size = 128
history = top_model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(valid_data, valid_labels))

top_model.save_weights(top_model_weights_path)

plot_training_curves(history.history);

predict_train = top_model.predict(train_data)

plot_confusion_matrix(train_labels, predict_train, macro_classes,                      
                      title="Train confusion matrix");
plt.figure()
plot_confusion_matrix(train_labels, predict_train, macro_classes,                      
                      title="Train confusion matrix",
                     normalize=True);

def compute_bottleneck_features(xs):
    xs = preprocess_input(xs.copy())
    return bottleneck_model.predict(xs)

rest_train_data = compute_bottleneck_features(X_train[train_subset:])
rest_train_labels = Y_train[train_subset:]

epochs = 50
batch_size = 128
history2 = top_model.fit(rest_train_data, rest_train_labels,
               epochs=epochs,
               batch_size=batch_size,
               validation_data=(valid_data, valid_labels))

from helpers import combine_histories
plot_training_curves(combine_histories(history.history, history2.history));

top_model.save_weights(top_model_weights_path)

predict_valid = top_model.predict(valid_data)

top_model.evaluate(valid_data, valid_labels)

plot_confusion_matrix(valid_labels, predict_valid, macro_classes,                      
                      title="Validation confusion matrix");
plt.figure()
plot_confusion_matrix(valid_labels, predict_valid, macro_classes,                      
                      title="Validation confusion matrix",
                     normalize=True);

