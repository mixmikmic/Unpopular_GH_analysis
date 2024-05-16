### Importing required python modules for these computations ####
# General stuff:
get_ipython().magic('matplotlib inline')
import numpy as np
import mkl
import matplotlib.pyplot as plt
from scipy import misc
import matplotlib.cm as cm

import sys
def isneeded(x):
    if x not in sys.path:
        sys.path.append(x)

isneeded('/Users/curt/openMSI_SVN/openmsi-tk/')
isneeded('/Users/curt/openMSI_localdata/')

#OpenMSI stuff for getting my source images:
from omsi.dataformat.mzml_file_CF import *

# Image registration
import imreg_dft as ird

#Reading a datafile of spotted samples in MALDI
omsi_ms2 = mzml_file(basename="/Users/curt/openMSI_localdata/re-reimaging_TI.mzML")

#Plotting TIC images for each scan type
f, axarr = plt.subplots(3, 1, figsize=(6, 9))
for ind in range(len(omsi_ms2.data)):
    axarr[ind].imshow(omsi_ms2.data[ind][:, :, :].sum(axis=2).T)  #total ion chromatogram images
    axarr[ind].set_title('TIC for scan type '+omsi_ms2.scan_types[ind])

#Plotting spectra at the most intense pixel for each scan type:
f, axarr = plt.subplots(3, 1, figsize=(9, 12))
for ind in range(len(omsi_ms2.data)):
    nx, ny, nmz = omsi_ms2.data[ind].shape
    maxx, maxy = np.unravel_index(omsi_ms2.data[ind][:, :, :].sum(axis=2).argmax(), dims=[nx, ny])
    axarr[ind].plot(omsi_ms2.data[ind][maxx, maxy, :])  
    axarr[ind].set_title(('Spectrum at pixel (%s, %s) for scan type ' % (maxx, maxy))+omsi_ms2.scan_types[ind])

#grayscale image of plate:
photo = misc.imread('/Users/curt/openMSI_localdata/Simple_canolaTAG_vs_sesameTAG_vs_blank_plate_image.bmp')
plt.figure(figsize=(20,10))
f = plt.imshow(photo, cmap=cm.Greys_r)

#plotting optical image resized to be the same size as MS image

ms1ticImage = omsi_ms2.data[1][:, :, :].sum(axis=2).T
photoSmall = misc.imresize(photo,ms1ticImage.shape)

f, ax = plt.subplots(2, 1)
ax[0].imshow(photoSmall, cmap=cm.Greys_r)
ax[1].imshow(ms1ticImage, cmap=cm.Greys_r)

#Normalize the MS image by the intensity of the triolein MS1 peak
mzind = abs(omsi_ms2.mz[1]-907.7738).argmin()
ms1_907 = omsi_ms2.data[1][:, :, mzind-2:mzind+2].sum(axis=2).T
normimage = ((ms1_907)) / (ms1ticImage+1.)
print normimage.dtype
f, ax = plt.subplots(2, 1)
ax[0].imshow(photoSmall, cmap=cm.Greys_r)
ax[1].imshow(normimage)

result = ird.similarity(photoSmall, normimage, numiter=20)
f, axarr = plt.subplots(2,2)
axarr[0, 0].imshow(photoSmall)
axarr[0, 1].imshow(normimage)
axarr[1, 0].imshow(photoSmall-normimage)
axarr[1, 1].imshow(result['timg'])
print result['tvec']

#Overlaying the "registered" MSI data on the optical image to make the final plot
msBig = misc.imresize(result['timg'], photo.shape, 'bicubic')

msBig_masked = np.ma.masked_where(msBig <= 35, msBig)

plt.figure(figsize=(30,15))
plt.imshow(photo, cmap=cm.Greys_r, alpha=1)
plt.imshow(msBig_masked, alpha=0.7)

