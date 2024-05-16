import tensorflow as tf
import numpy as np
from astropy.io import fits
import importlib.util
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

lens_band1 = fits.open('samples/ground_based/lens/Band1/imageSDSS_R-100002.fits')[0].data
lens_band2 = fits.open('samples/ground_based/lens/Band2/imageSDSS_I-100002.fits')[0].data
lens_band3 = fits.open('samples/ground_based/lens/Band3/imageSDSS_G-100002.fits')[0].data
lens_band4 = fits.open('samples/ground_based/lens/Band4/imageSDSS_U-100002.fits')[0].data
lens = np.stack([lens_band1, lens_band2, lens_band3, lens_band4], 2)

# show the green band
plt.imshow(lens[:,:,2])

nolens_band1 = fits.open('samples/ground_based/nolens/Band1/imageSDSS_R-100004.fits')[0].data
nolens_band2 = fits.open('samples/ground_based/nolens/Band2/imageSDSS_I-100004.fits')[0].data
nolens_band3 = fits.open('samples/ground_based/nolens/Band3/imageSDSS_G-100004.fits')[0].data
nolens_band4 = fits.open('samples/ground_based/nolens/Band4/imageSDSS_U-100004.fits')[0].data
nolens = np.stack([nolens_band1, nolens_band2, nolens_band3, nolens_band4], 2)

# show the green band
plt.imshow(nolens[:,:,2])

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

graph, nn = load_arch("arch_baseline.py", 4)

images = nn.prepare(np.array([lens, nolens]))

sess = tf.Session(graph=graph)

sess.run(tf.variables_initializer(graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

nn.predict(sess, images)

def load_backup(sess, graph, backup):
    with graph.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, backup)

load_backup(sess, graph, 'trained_variables/ground_based/baseline')

nn.predict(sess, images)

images = (images - images.mean()) / images.std()

nn.predict(sess, images)



