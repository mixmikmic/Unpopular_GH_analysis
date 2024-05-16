import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.misc import toimage
from  PIL import Image

plt.rcParams['figure.figsize'] = (20.0, 10.0)

get_ipython().run_line_magic('matplotlib', 'inline')

# list the image folders

path = 'C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\images'

img_folder = os.listdir(path)

print('{} folders in img_folder'.format(len(img_folder)))
print('\n'.join(img_folder))

pug_folder = os.path.join(path,'n02110958-pug')

print('{} images in pug_folder'.format(len(pug_folder)))
pug_images = os.listdir(pug_folder)

os.chdir(pug_folder)

im = Image.open(pug_images[12],'r')
plt.imshow(im)

from __future__ import print_function

print(im.format, im.size, im.mode)

im = Image.open(pug_images[13],'r')
plt.imshow(im)
print(im.format, im.size, im.mode)

im = Image.open(pug_images[68],'r')
plt.imshow(im)
print(im.format, im.size, im.mode)

# check for class imbalance of labels and data

file_list = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\file_list.mat''')
print(file_list.keys())

df = pd.DataFrame(file_list['labels'], columns=['breed_label'], index=file_list['file_list'], dtype='int64')
df.head()

file_list['annotation_list'][0][0]
#annotation simply provides the image file name per label breed..same as file_list

df.shape

df.info()

# checking for missing labels
df['breed_label'].isnull().any()

missing_values_count = df.isnull().sum()
missing_values_count

df.nunique()

avg_num_images = 20580/120
median_num_images = df['breed_label'].value_counts().median()

# given 20,580 labels, does this correspond to the amount of images?

image_count = sum([len(files) for r, d, files in os.walk(path)])

print(image_count)


df.hist(bins=120, grid=False, figsize=(15,5))
plt.axhline(avg_num_images, color='r', linestyle='dashed', linewidth=2, label='Average # of Images')
plt.axhline(median_num_images, color='g', linestyle='dashed', linewidth=2, label='Median # of Images')
plt.legend()
plt.title('Count of Images per Breed Label')

plt.show()

#repeat the process with loading and referencing images and display in subplot/grid
# currently getting Errno13, permission denied. Skip for now...
"""
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        for img in os.listdir(filename):
            img = Image.open(os.path.join(folder, filename), mode='r')
            images.append(img)
    return images

image_set = load_images_from_folder(path)
"""
#root_folder = '[whatever]/data/train'
# use path variable
#folders = [os.path.join(path, x) for x in img_folder]
#all_images = [img for folder in folders for img in load_images_from_folder(folder)]

train_data = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\train_data.mat''')
print(train_data.keys())

train_data



train_data['train_fg_data'][0]
print(len(train_data['train_fg_data'][0]))

train_data['train_data'][0]
print(len(train_data['train_data'][0]))

#what exactly is the train_info values comprised of?
train_data['train_info']

len(train_data['train_info'])
#so it's a single array... of multiple arrays

file_list = train_data['train_info'][0][0][0] #array of array of arrays...
annotation_list = train_data['train_info'][0][0][1]
labels = train_data['train_info'][0][0][2]
fg_ids = train_data['train_info'][0][0][3]

print(file_list.shape)
print(annotation_list.shape)
print(labels.shape)
print(fg_ids.shape)

print(labels[0][0].shape)

train_info = pd.DataFrame(train_data['train_info'][0])
train_info.head()



#define NP arrays to construct DF will need to do this for the test data too

def mat_to_df(filepath, struc_label):
    matfile = loadmat(filepath)
    file_list_arr = matfile[struc_label][0][0][0]
    annot_list_arr = matfile[struc_label][0][0][1]
    labels_arr = matfile[struc_label][0][0][2]
    fg_id_arr = matfile[struc_label][0][0][3]
    
    data = np.array([annot_list_arr, labels_arr, fg_id_arr])
    
    df = pd.DataFrame(data)
    return df

#train_data = mat_to_df(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\train_data.mat''', 'train_info')
#test_data = mat_to_df(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\test_data.mat''', 'test_info')

'''
def print_mat_nested(d, indent=0, nkeys=0):
    """Pretty print nested structures from .mat files   
    Inspired by: `StackOverflow <http://stackoverflow.com/questions/3229419/pretty-printing-nested-dictionaries-in-python>`_
    """
    # Subset dictionary to limit keys to print.  Only works on first level
   
    if nkeys>0:
        d = {k: d[k] for k in d.keys()[:nkeys]} # Dictionary comprehension: limit to first nkeys keys.
    
    if isinstance(d, dict): 
        for key, value in d.iteritems(): # iteritems loops through key, value pairs
            print('\t' * indent + 'Key: ' + str(key)) 
            print_mat_nested(value, indent+1)
            
    if isinstance(d,np.ndarray) and d.dtype.names is not None:
        for n in d.dtype.names:  
            print('\t' * indent + 'Field: ' + str(n))
            print_mat_nested(d[n], indent+1)
''' 
#not used

#I realized I did not unzip/unpack a file which contains the list of stratified train/test splits.

train_list = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\train_list.mat''')['file_list']
test_list = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\test_list.mat''')['file_list']

len(train_list)

len(test_list)

#creating mock variables for TF

train_labels = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\train_list.mat''')['labels']
test_labels = loadmat(r'''C:\\Users\\Garrick\\Documents\\Springboard\\Capstone Project 2\\datasets\\train_list.mat''')['labels']


