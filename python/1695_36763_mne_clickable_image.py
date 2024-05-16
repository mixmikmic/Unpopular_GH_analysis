from scipy.ndimage import imread
import numpy as np
from matplotlib import pyplot as plt
import sys
import pandas as pd
sys.path.insert(0, '/Users/choldgraf/github/mne-python/')
import mne
from mne.viz.utils import ClickableImage
from mne.channels.layout import generate_2d_layout

plt.rcParams['image.cmap'] = 'gray'

im_path = '/Users/choldgraf/github/mne-python/mne/data/image/mni_brain.gif'
layout_path = '/Users/choldgraf/github/mne-python/mne/data/image/custom_layout.lay'

im = imread(im_path)

# Make sure that inline plotting is off before clicking
get_ipython().magic('matplotlib qt')
click = ClickableImage(im)

get_ipython().magic('matplotlib inline')

# The click coordinates are stored as a list of tuples
click.plot_clicks()
coords = click.coords
print coords

# Generate a layout from our clicks and normalize by the image
# lt = generate_2d_layout(np.vstack(coords), bg_image=im) 
# lt.save(layout_path + 'custom_layout.lay')  # To save if we want

# Or if we've already got the layout, load it
lt = mne.channels.read_layout(layout_path)

# Create some fake data
nchans = len(coords)
nepochs = 50
sr = 1000
nsec = 5
events = np.arange(nepochs).reshape([-1, 1])
events = np.hstack([events, np.zeros([nepochs, 2])])
data = np.random.randn(nepochs, nchans, sr * nsec)
info = mne.create_info(nchans, sr, ch_types='eeg')
epochs = mne.EpochsArray(data, info, events)

# Using the native plot_topo function
f = mne.viz.plot_topo(epochs.average(), layout=lt)

# Now with the image plotted in the background
f = mne.viz.plot_topo(epochs.average(), layout=lt)
ax = f.add_axes([0, 0, 1, 1])
ax.imshow(im)
ax.set_zorder(-1)

