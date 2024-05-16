import tensorflow as tf
import numpy as np
import glob
import os
from astropy.io import fits
import importlib.util
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

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

graph, nn = load_arch("arch_views.py", 4)

def load_backup(sess, graph, backup):
    with graph.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, backup)

sess = tf.Session(graph=graph)
os.system('cd trained_variables/ground_based/ && 7za e views.7z.001')
load_backup(sess, graph, 'trained_variables/ground_based/views')

files = sorted(glob.glob('../LE-data/GroundBasedTraining/npz/*.npz'))[:3000]

concat = []
for i in range(0, len(files), 50):
    images = [np.load(x)['image'] for x in files[i: i + 50]]
    images = nn.prepare(np.array(images))
    res = sess.run(nn.embedding_input, feed_dict={nn.tfx: images})
    concat.append(res)
    print(i, end=' ')
concat = np.array(concat).reshape((3000, 4))

plt.hist(concat[:,0], bins=150);

plt.hist(concat[:,1], bins=150);

plt.hist(concat[:,2], bins=150);

plt.hist(concat[:,3], bins=150);



