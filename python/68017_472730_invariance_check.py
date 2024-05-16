import tensorflow as tf
import numpy as np
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

graph_baseline, nn_baseline = load_arch("arch_baseline.py", 1)
graph_invariant, nn_invariant = load_arch("arch_invariant.py", 1)

sess_baseline = tf.Session(graph=graph_baseline)
sess_invariant = tf.Session(graph=graph_invariant)

sess_baseline.run(tf.variables_initializer(graph_baseline.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
sess_invariant.run(tf.variables_initializer(graph_invariant.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

image = fits.open('samples/space_based/lens/imageEUC_VIS-100002.fits')[0].data
plt.imshow(image)
image = nn_baseline.prepare(image.reshape(101, 101, 1))

def dihedral(x, i):
    x = x.copy()
    if i & 4:
        x = np.transpose(x, (1, 0, 2))  # tau[4]
    if i & 1:
        x = x[:, ::-1, :]  # tau[1]
    if i & 2:
        x = x[::-1, :, :]  # tau[2]
    return x

images = np.array([dihedral(image, i) for i in range(8)])

ps_baseline = sess_baseline.run(nn_baseline.tfp, feed_dict={nn_baseline.tfx: images})
plt.plot(ps_baseline)
print(ps_baseline)

ps_invariant = sess_invariant.run(nn_invariant.tfp, feed_dict={nn_invariant.tfx: images})
plt.plot(ps_invariant)
print(ps_invariant)

test = sess_invariant.run(nn_invariant.test, feed_dict={nn_invariant.tfx: images})
test = np.reshape(test, (8, 8, -1))

step = test[0].max() - test[0].min()
for i in range(8):
    plt.plot(test[i].flatten() + step * i)

mt = np.array([ [0, 1, 2, 3, 4, 5, 6, 7], [1, 0, 3, 2, 5, 4, 7, 6],
                [2, 3, 0, 1, 6, 7, 4, 5], [3, 2, 1, 0, 7, 6, 5, 4],
                [4, 6, 5, 7, 0, 2, 1, 3], [5, 7, 4, 6, 1, 3, 0, 2],
                [6, 4, 7, 5, 2, 0, 3, 1], [7, 5, 6, 4, 3, 1, 2, 0]])
# tau[mt[a,b]] = tau[a] o tau[b]

iv = np.array([0, 1, 2, 3, 4, 6, 5, 7])
# tau[iv[a]] is the inverse of tau[a]

for i in range(8):
    plt.plot(test[i][mt[i]].flatten() + step * i)



