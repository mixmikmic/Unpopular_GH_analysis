get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt

path = '/home/darwin/Projects/datasets/packed/f01906891b1f8ea2e2715d9f99f126f6.npz'

archive = np.load(path)
print('keys:', archive.files)

images = archive['images']
offsets = archive['offsets']

print('images.shape:', images.shape)
print('offsets.shape:', offsets.shape)

sample_idx = 2335

print('sample images shape:', images[sample_idx].shape)

a=np.split(images,48)
print(a[0].shape)

img = np.split(images[sample_idx], 2, axis=-1)
plt.imshow(img[0].squeeze(), cmap='gray')

plt.imshow(img[1].squeeze(), cmap='gray')

# The order (although it doesn't matter) is: top-left, bottom-left, bottom-right, top-right
print(offsets[sample_idx])

# Efficient loading in Keras using a Python generator

import os.path
import glob

import numpy as np

def data_loader(path, batch_size=64):
    """Generator to be used with model.fit_generator()"""
    while True:
        for npz in glob.glob(os.path.join(path, '*.npz')):
            # Load pack into memory
            archive = np.load(npz)
            images = archive['images']
            offsets = archive['offsets']
            # Yield minibatch
            for i in range(0, len(offsets), batch_size):
                end_i = i + batch_size
                try:
                    batch_images = images[i:end_i]
                    batch_offsets = offsets[i:end_i]
                except IndexError:
                    continue
                # Normalize
                batch_images = (batch_images - 127.5) / 127.5
                batch_offsets = batch_offsets / 32.
                yield batch_images, batch_offsets

# Dataset-specific
train_data_path = '/path/to/training-data'
test_data_path = '/path/to/test-data'
num_samples = 150 * 3072 # 158 archives x 3,072 samples per archive, but use just 150 and save the 8 for testing

# From the paper
batch_size = 64
total_iterations = 90000

steps_per_epoch = num_samples / batch_size # As stated in Keras docs
epochs = int(total_iterations / steps_per_epoch)

# model is some Keras Model instance

# Train
model.fit_generator(data_loader(train_data_path, batch_size),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs)

# Test
model.evaluate_generator(data_loader(test_data_path, 1),
                         steps=100)



