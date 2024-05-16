import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import keras
from keras.layers import Input, Embedding, LSTM, Dense, Conv2D
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

IMAGE_HEIGHT, IMAGE_WIDTH = 240, 240
TUMOR_IMAGES = 155
AGE_CLASSES = 100
MAX_SURVIVAL = 2000
df = pd.read_csv("data/survival_data.csv")
df.head()

X = np.load('data/tumors_nz.npy')
X = X[:, 14:143, :, :, :] # All images from 0 to 15 and 144 to the end are totally black
Y = df['Survival']

plt.imshow(X[100, 77, 0, :, :], cmap='gray')
np.unique(X[100, 77, 0, :, :])

def get_splitted_data(X, ages, labels):
    assert len(X) == len(labels) == len(ages)
    # OneHotEncoding for ages
    enc_table = np.eye(AGE_CLASSES)
    ages_ohe = np.array([enc_table[int(round(x))] for x in ages])
    # Normalize labels
    labels /= MAX_SURVIVAL
    # Split data into: 70% train, 15% test, 15% validation
    cuts = [int(.70*len(X)), int(.85*len(X))]
    X1_train, X1_test, X1_val = np.split(X, cuts)
    X2_train, X2_test, X2_val = np.split(ages_ohe, cuts)
    Y_train, Y_test, Y_val = np.split(labels, cuts)
    return X1_train, X2_train, Y_train, X1_test, X2_test, Y_test, X1_val, X2_val, Y_val

X1_train, X2_train, Y_train, X1_test, X2_test, Y_test, X1_val, X2_val, Y_val = get_splitted_data(X, df['Age'], df['Survival'])

with tf.device('/gpu:0'):
    
    main_input = Input(shape=X.shape[1:], dtype='float32', name='main_input')
    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(main_input)
    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 1, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2, 1, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    cnn_out = Dense(64, activation='relu')(x)
    # auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(cnn_out)
    
    auxiliary_input = Input(shape=(AGE_CLASSES,), name='aux_input', dtype='float32')
    x = keras.layers.concatenate([cnn_out, auxiliary_input])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)
    
    
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])  # , auxiliary_output
    
    # RMSprop uses:
    # - Momentum taking knowledge from previous steps into account about where
    #   we should be heading (prevents gradient descent to oscillate)
    # - Uses recent gradients to adjust alpha
    #   (when the gradient is very large, alpha is reduced and vice-versa)
    # Later we should test if AdaDelta or Adam are improving our results (quite similar to RMSprop)
    model.compile(optimizer='Adam',
                  metrics=['accuracy'],
                  loss={'main_output': 'mean_squared_error'},  # 'aux_output': 'mean_squared_error'
                  loss_weights={'main_output': 1.})  # , 'aux_output': 0.2

    # And trained it via:
    model.fit({'main_input': X1_train, 'aux_input': X2_train},
              {'main_output': Y_train},  # 'aux_output': Y
              epochs=50, batch_size=32, verbose=1,
              validation_data=({'main_input': X1_test, 'aux_input': X2_test}, Y_test))

for i in range(5):
    prediction = int(model.predict({'main_input': X[i:i+1], 'aux_input': ages_ohe[i:i+1]}) * MAX_SURVIVAL)
    print('Patient: {}, Age: {}, GT: {}, Prediction: {}'.format(df.values[i, 0], df.values[i, 1], df.values[i, 2], prediction))

