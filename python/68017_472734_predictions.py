import os
import tensorflow as tf
import numpy as np
from astropy.io import fits
import importlib.util

def load_module(path):
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_arch(arch_path, bands):
    arch = load_module(arch_path)
    nn = arch.CNN()

    g = tf.Graph()
    with g.as_default():
        nn.create_architecture(bands=bands)
    return g, nn

def load_backup(sess, graph, backup):
    with graph.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, backup)

graph, nn = load_arch("arch_invariant.py", 1)
sess = tf.Session(graph=graph)
load_backup(sess, graph, 'trained_variables/space_based/invariant')

path = 'samples/space_based/lens/'
images = [fits.open(os.path.join(path, file))[0].data for file in os.listdir(path)]

path = 'samples/space_based/nolens/'
images += [fits.open(os.path.join(path, file))[0].data for file in os.listdir(path)]

images = nn.prepare(np.array(images).reshape((-1, 101, 101, 1)))

predictions = nn.predict(sess, images)
predictions

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

fig = plt.figure(figsize=(10,3))

for i in range(len(images)):
    a = fig.add_subplot(2, len(images) // 2,i+1)
    img = a.imshow(images[i, :, :, 0])
    img.set_cmap('hot')
    a.axis('off')
    a.annotate('p={:.3f}'.format(predictions[i]), xy=(10,80), color='white')

plt.subplots_adjust(left=0, bottom=0, top=1, right=1, wspace=0, hspace=0)



