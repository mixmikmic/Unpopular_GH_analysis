import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os
os.chdir('..')

import sys
sys.path.insert(0, './python')
import caffe


import os
import h5py
import shutil
import tempfile

import sklearn
import sklearn.datasets
import sklearn.linear_model

import pandas as pd

X, y = sklearn.datasets.make_classification(
    n_samples=10000, n_features=4, n_redundant=0, n_informative=2, 
    n_clusters_per_class=2, hypercube=False, random_state=0
)

# Split into train and test
X, Xt, y, yt = sklearn.cross_validation.train_test_split(X, y)

# Visualize sample of the data
ind = np.random.permutation(X.shape[0])[:1000]
df = pd.DataFrame(X[ind])
_ = pd.scatter_matrix(df, figsize=(9, 9), diagonal='kde', marker='o', s=40, alpha=.4, c=y[ind])

get_ipython().run_cell_magic('timeit', '', "# Train and test the scikit-learn SGD logistic regression.\nclf = sklearn.linear_model.SGDClassifier(\n    loss='log', n_iter=1000, penalty='l2', alpha=5e-4, class_weight='auto')\n\nclf.fit(X, y)\nyt_pred = clf.predict(Xt)\nprint('Accuracy: {:.3f}'.format(sklearn.metrics.accuracy_score(yt, yt_pred)))")

# Write out the data to HDF5 files in a temp directory.
# This file is assumed to be caffe_root/examples/hdf5_classification.ipynb
dirname = os.path.abspath('./examples/hdf5_classification/data')
if not os.path.exists(dirname):
    os.makedirs(dirname)

train_filename = os.path.join(dirname, 'train.h5')
test_filename = os.path.join(dirname, 'test.h5')

# HDF5DataLayer source should be a file containing a list of HDF5 filenames.
# To show this off, we'll list the same data file twice.
with h5py.File(train_filename, 'w') as f:
    f['data'] = X
    f['label'] = y.astype(np.float32)
with open(os.path.join(dirname, 'train.txt'), 'w') as f:
    f.write(train_filename + '\n')
    f.write(train_filename + '\n')
    
# HDF5 is pretty efficient, but can be further compressed.
comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
with h5py.File(test_filename, 'w') as f:
    f.create_dataset('data', data=Xt, **comp_kwargs)
    f.create_dataset('label', data=yt.astype(np.float32), **comp_kwargs)
with open(os.path.join(dirname, 'test.txt'), 'w') as f:
    f.write(test_filename + '\n')

from caffe import layers as L
from caffe import params as P

def logreg(hdf5, batch_size):
    # logistic regression: data, matrix multiplication, and 2-class softmax loss
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    n.ip1 = L.InnerProduct(n.data, num_output=2, weight_filler=dict(type='xavier'))
    n.accuracy = L.Accuracy(n.ip1, n.label)
    n.loss = L.SoftmaxWithLoss(n.ip1, n.label)
    return n.to_proto()

train_net_path = 'examples/hdf5_classification/logreg_auto_train.prototxt'
with open(train_net_path, 'w') as f:
    f.write(str(logreg('examples/hdf5_classification/data/train.txt', 10)))

test_net_path = 'examples/hdf5_classification/logreg_auto_test.prototxt'
with open(test_net_path, 'w') as f:
    f.write(str(logreg('examples/hdf5_classification/data/test.txt', 10)))

from caffe.proto import caffe_pb2

def solver(train_net_path, test_net_path):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and test networks.
    s.train_net = train_net_path
    s.test_net.append(test_net_path)

    s.test_interval = 1000  # Test after every 1000 training iterations.
    s.test_iter.append(250) # Test 250 "batches" each time we test.

    s.max_iter = 10000      # # of times to update the net (training iterations)

    # Set the initial learning rate for stochastic gradient descent (SGD).
    s.base_lr = 0.01        

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 5000

    # Set other optimization parameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- just once at the end of training.
    # For larger networks that take longer to train, you may want to set
    # snapshot < max_iter to save the network and training state to disk during
    # optimization, preventing disaster in case of machine crashes, etc.
    s.snapshot = 10000
    s.snapshot_prefix = 'examples/hdf5_classification/data/train'

    # We'll train on the CPU for fair benchmarking against scikit-learn.
    # Changing to GPU should result in much faster training!
    s.solver_mode = caffe_pb2.SolverParameter.CPU
    
    return s

solver_path = 'examples/hdf5_classification/logreg_solver.prototxt'
with open(solver_path, 'w') as f:
    f.write(str(solver(train_net_path, test_net_path)))

get_ipython().run_cell_magic('timeit', '', 'caffe.set_mode_cpu()\nsolver = caffe.get_solver(solver_path)\nsolver.solve()\n\naccuracy = 0\nbatch_size = solver.test_nets[0].blobs[\'data\'].num\ntest_iters = int(len(Xt) / batch_size)\nfor i in range(test_iters):\n    solver.test_nets[0].forward()\n    accuracy += solver.test_nets[0].blobs[\'accuracy\'].data\naccuracy /= test_iters\n\nprint("Accuracy: {:.3f}".format(accuracy))')

get_ipython().system('./build/tools/caffe train -solver examples/hdf5_classification/logreg_solver.prototxt')

from caffe import layers as L
from caffe import params as P

def nonlinear_net(hdf5, batch_size):
    # one small nonlinearity, one leap for model kind
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    # define a hidden layer of dimension 40
    n.ip1 = L.InnerProduct(n.data, num_output=40, weight_filler=dict(type='xavier'))
    # transform the output through the ReLU (rectified linear) non-linearity
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    # score the (now non-linear) features
    n.ip2 = L.InnerProduct(n.ip1, num_output=2, weight_filler=dict(type='xavier'))
    # same accuracy and loss as before
    n.accuracy = L.Accuracy(n.ip2, n.label)
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()

train_net_path = 'examples/hdf5_classification/nonlinear_auto_train.prototxt'
with open(train_net_path, 'w') as f:
    f.write(str(nonlinear_net('examples/hdf5_classification/data/train.txt', 10)))

test_net_path = 'examples/hdf5_classification/nonlinear_auto_test.prototxt'
with open(test_net_path, 'w') as f:
    f.write(str(nonlinear_net('examples/hdf5_classification/data/test.txt', 10)))

solver_path = 'examples/hdf5_classification/nonlinear_logreg_solver.prototxt'
with open(solver_path, 'w') as f:
    f.write(str(solver(train_net_path, test_net_path)))

get_ipython().run_cell_magic('timeit', '', 'caffe.set_mode_cpu()\nsolver = caffe.get_solver(solver_path)\nsolver.solve()\n\naccuracy = 0\nbatch_size = solver.test_nets[0].blobs[\'data\'].num\ntest_iters = int(len(Xt) / batch_size)\nfor i in range(test_iters):\n    solver.test_nets[0].forward()\n    accuracy += solver.test_nets[0].blobs[\'accuracy\'].data\naccuracy /= test_iters\n\nprint("Accuracy: {:.3f}".format(accuracy))')

get_ipython().system('./build/tools/caffe train -solver examples/hdf5_classification/nonlinear_logreg_solver.prototxt')

# Clean up (comment this out if you want to examine the hdf5_classification/data directory).
shutil.rmtree(dirname)

