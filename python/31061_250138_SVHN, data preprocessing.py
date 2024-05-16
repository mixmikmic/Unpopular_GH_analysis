# change theme of ipython notebooks
from IPython.core.display import HTML
import urllib2
HTML(urllib2.urlopen('http://bit.ly/1Bf5Hft').read())

# step 1
import numpy as np
import cPickle as pickle
import h5py

#f = h5py.File('train/digitStruct.mat')
f = h5py.File('test/digitStruct.mat')

metadata= {}
metadata['height'] = []
metadata['label'] = []
metadata['left'] = []
metadata['top'] = []
metadata['width'] = []

def print_attrs(name, obj):
    vals = []
    if obj.shape[0] == 1:
        vals.append(int(obj[0][0]))
    else:
        for k in range(obj.shape[0]):
            vals.append(int(f[obj[k][0]][0][0]))
    metadata[name].append(vals)

for item in f['/digitStruct/bbox']:
    f[item[0]].visititems(print_attrs)
    
#with open('train_metadata.pickle','wb') as pf:
with open('test_metadata.pickle','wb') as pf:
    pickle.dump(metadata, pf, pickle.HIGHEST_PROTOCOL)    

# check the number of digits
image_num = 33402 # 33402 for train, 13068 for test
count = 0
for i in range(image_num):                
    digit_num += len(metadata['width'][i])  
print count

# step 2 

import cPickle as pickle
#with open('train_metadata.pickle', 'rb') as f:
with open('test_metadata.pickle', 'rb') as f:  
    metadata = pickle.load(f)

import numpy as np 
import cv2
#image_num = 33402
#sample_num = 73257
image_num =  13068
sample_num = 26032

dataset = np.ndarray(shape=(sample_num, 28, 28),dtype=np.float32)
lables = np.ndarray(shape=(sample_num, ),dtype=np.int)

def crop(image, i,j):
    top = metadata['top'][i][j]
    height = metadata['height'][i][j]
    left = metadata['left'][i][j]
    width = metadata['width'][i][j]
    if left < 0:
        left, width = 0, width+left
  
    return image[top:top+height, left:left+width]

depth = 255.0  # pixel depth
for i in range(image_num):
    #path = 'train/{0}.png'.format(i+1)
    path = 'test/{0}.png'.format(i+1)
    image = cv2.imread(path)
    num = len(metadata['width'][i])  
    for j in range(num):
        crop_image = crop(image,i,j)
        gray_image = rgb2gray(crop_image)
        #print i,j  # find (250,0) has left value of -1
        resize_image = cv2.resize(gray_image,(28,28))
        normal_image = resize_image/depth -0.5

        dataset[count,:,:] = normal_image
        lables[count] = metadata['label'][i][j] % 10

#with open('train_dataset_labels.pickle','wb') as pf:
with open('test_dataset_labels.pickle','wb') as pf:  
    pickle.dump((dataset,lables), pf, pickle.HIGHEST_PROTOCOL)  
  
del metadata # clean cache

import numpy as np
import scipy.io
# train_mat = scipy.io.loadmat('train_32x32.mat')  # dict.key() ['y', 'X', '__version__', '__header__', '__globals__']

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def mat2data(matfile):
    mat = scipy.io.loadmat(matfile)
    Xdata = mat['X']
    ydata = mat['y']
    size = Xdata.shape   # (32,32,3,73257)
    print 'size = {0}'.format(size)
    image_size_x = size[0]
    image_size_y = size[0]
    num_samples = size[3]

    depth = 255.0  # pixel depth
    dataset = np.ndarray(shape=(num_samples, image_size_x, image_size_y),dtype=np.float32)
    labels = np.ndarray(shape=(num_samples,),dtype=np.int8)
    for i in range(num_samples):
        dataset[i,:,:] = rgb2gray(Xdata[:,:,:,i]) /depth - 0.5  # 3D-2D and normalize
        labels[i] = ydata[i][0] % 10
        return dataset, labels

X_train,y_train = mat2data('train_32x32.mat')
X_test, y_test  = mat2data('test_32x32.mat') 
with open('train_test_32x32.pickle','wb') as pf:
    pickle.dump(((X_train,y_train),(X_test, y_test)), pf, pickle.HIGHEST_PROTOCOL)  

