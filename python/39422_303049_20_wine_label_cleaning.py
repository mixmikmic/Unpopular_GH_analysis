import multiprocess as mp

from glob import glob
import re
import pandas as pd
import numpy as np
import os

from PIL import Image
import cv2

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

snooth_images = glob('../priv/images/snooth_dot_com_*.*')
wine_dot_com_images = glob('../priv/images/wine_dot_com_*.*')

int_sorter = lambda x: int(re.search(r"""_([0-9]+)\.""", x).group(1))
snooth_images = sorted(snooth_images, key=int_sorter)
wine_dot_com_images = sorted(wine_dot_com_images, key=int_sorter)

def get_sizes(file_list):
    
    file_df = list()

    for fil in file_list:
        try:
            with Image.open(fil) as im:
                width, height = im.size        
        except:
            width = np.NaN
            height = np.NaN

        file_ser = pd.Series({'image_name':fil, 'width':width, 'height':height})
        
        file_df.append(file_ser)
        
    return file_df

file_list = snooth_images
file_list.extend(wine_dot_com_images)

nthreads = 48
pool = mp.Pool(processes=nthreads)
size_list = pool.map(get_sizes, np.array_split(file_list, nthreads))
pool.close()

image_size_df = pd.concat(sum(size_list,[]), axis=1).T

image_size_df['height'] = image_size_df.height.astype(int)
image_size_df['width'] = image_size_df.width.astype(int)
image_size_df['area'] = image_size_df.height * image_size_df.width

image_size_df.shape

def extract_basename(x):
    return os.path.splitext(os.path.basename(x))[0]

image_size_df['basename'] = image_size_df.image_name.apply(extract_basename)

image_size_df.area.min(), image_size_df.area.max()

image_size_df.hist('area', bins=100)

image_size_df.hist('height', bins=100)

image_size_df.hist('width', bins=100)

img = Image.open('../priv/images/snooth_dot_com_47220.png')
img.size

plt.imshow(img)

plt.imshow(Image.open('../priv/images/wine_dot_com_8516.jpg'))

mask = (image_size_df.area>1.0e6)&(image_size_df.area<1.5e6)
mask.sum()

image_size_df.loc[mask].to_pickle('../priv/pkl/20_wine_label_analysis_large_labels.pkl')

image_size_df.head()

mask = image_size_df.area>10000
image_size_df_out = image_size_df[mask]
image_size_df_out.shape

mask = np.invert((image_size_df.area>1.5e6)&(image_size_df['basename'].str.contains('wine_dot_com')))
image_size_df_out = image_size_df_out[mask]
image_size_df_out.shape

image_size_df_out.to_pickle('../priv/pkl/20_wine_label_analysis_all_labels.pkl')

image = np.asarray(Image.open('../priv/images/wine_dot_com_8516.jpg'))
height, width = image.shape[:2]
nrows = 3
ncols = 3
w = int(width/ncols)
h = int(height/nrows)

segments = list()
for r in range(nrows):
    for c in range(ncols):
        x_beg = c*w
        y_beg = r*h
        
        if c != (ncols-1):
            x_end = (c+1)*w
        else:
            x_end = width+1
            
        if r != (nrows-1):
            y_end = (r+1)*h
        else:
            y_end = height+1
            
        segments.append((x_beg, x_end, y_beg, y_end))
        
segments

bins = (4, 6, 3)
# bins = (64, 64, 64)

image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
features = []

# loop over the segments
for (x_beg, x_end, y_beg, y_end) in segments:
    
    # construct a mask for each part of the image
    squareMask = np.zeros(image.shape[:2], dtype='uint8')
    cv2.rectangle(squareMask, (x_beg, y_beg), (x_end, y_end), 255, -1)
    
    hist = cv2.calcHist([image], [0, 1, 2], squareMask, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist).flatten()
    
    features.extend(hist)

f, axList = plt.subplots(nrows=3, ncols=3)

for his,ax in zip(hist, axList.flatten()):
    ax.hist(his, bins=25)

