import mne
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import ecogtools as et
import seaborn as sns
sns.set_style('white')

# This ensures that you'll be able to use interactive plots
get_ipython().magic('matplotlib notebook')

# These are stored as "fif" files, which MNE reads easily
brain = mne.io.Raw('./data/ecog_clean_raw.fif', add_eeg_ref=False, preload=True)
audio = mne.io.Raw('./data/spectrogram.fif', add_eeg_ref=False, preload=True)

# First we'll just glance through the brain activity.
# For plot visualizations
scale = brain._data[0].max()

# This should pop out an interactive plot, scroll through the data
f = brain.plot(scalings={'eeg': scale}, show=False)
f.set_size_inches(8, 5)

# Load our times file, which is stored as a CSV of events
mtime = pd.read_csv('./data/time_info.csv', index_col=0)

# Pull only the trials we care about
mtime = mtime.query('t_type=="mid" and t_num > 0')

# These are the start/stop times for sound
mtime.head()

# We will create an "events" object by turning the start times into indices
# Then turning it into an array of shape (n_events, 3)
ev = mtime['start'] * brain.info['sfreq']  # goes from seconds to index
ev = np.vstack([ev, np.zeros_like(ev), np.ones_like(ev)]).astype(int).T

# This is just a metadata dictionary.
# If we had multiple event types, we'd specify here.
einfo = dict(myevent=1)

# First columns == event start ix, last columns == event id
ev[:5]

# First low-pass filter to remove some noise
brain_epochs = brain.copy()
brain_epochs.filter(0, 20)

# Now we'll turn the raw array into epochs shape
tmin, tmax = -.5, 2
epochs = mne.Epochs(brain_epochs, ev, einfo, tmin, tmax,
                    baseline=(None, 0), preload=True)

# shape (trials, channels, time)
print(epochs._data.shape)

# Let's average across all epochs to see if any channels are responsive
epochs.average().plot()

# We'll do this on the raw data for reasons we can talk about later
brain.filter(70, 150)

# We'll also add zeros to our data so that it's of a length 2**N.
# In signal processing, everything goes faster if your data is length 2**N :-)
next_pow2 = int(np.ceil(np.log2(brain.n_times)))
brain.apply_hilbert(range(len(brain.ch_names)), envelope=True,
                    n_fft=2**next_pow2)

# Now that we've extracted the amplitude, we'll low-pass filter it to remove noise
brain.filter(None, 20)

# Now take another look at the data
scale = brain._data[0].max()
brain.plot(scalings={'eeg': scale})

tmin, tmax = -.5, 2
epochs = mne.Epochs(brain, ev, einfo, tmin, tmax,
                    baseline=(None, 0), preload=True)

# Note it's the same shape as before
print(epochs._data.shape)

# We'll rescale the epochs to show the increase over baseline using a
# "z" score. This subtracts the baseline mean, and divides by baseline
# standard deviation
_ = mne.baseline.rescale(epochs._data, epochs.times, [-.5, 0], 'zscore', copy=False)

# Let's look at the average plots again
epochs.average().plot()

# Use the arrow keys to move around.
# Green line == time 0. Dotted line == epoch start
# See if some electrodes seem to "turn on" in response to the sound.
scale = 10
f = epochs.plot(scalings=scale, n_epochs=10)
f.set_size_inches(8, 5)

# Another way to look at this is with an image.
# Here are the trial activations for one electrode:
# It looks like this channel is responsive
use_chan = 'TG37'
ix_elec = mne.pick_channels(epochs.ch_names, [use_chan])[0]
plt_elec = epochs._data[:, ix_elec, :]

f, ax = plt.subplots()
ax.imshow(plt_elec, aspect='auto', cmap=plt.cm.RdBu_r, vmin=-5, vmax=5)
f

f, ax = plt.subplots()

# This will plot 10 seconds.
ax.imshow(audio._data[:, :10*audio.info['sfreq']],
          aspect='auto', origin='lower', cmap=plt.cm.Reds)

# First we'll cut up the data so that we don't overload memory
# We will create an "events" object by turning the start times into indices
# Then turning it into an array of shape (n_events, 3)
ev = mtime['start'] * audio.info['sfreq']
ev = np.vstack([ev, np.zeros_like(ev), np.ones_like(ev)]).astype(int).T
einfo = dict(myevent=1)

# Now we'll turn the raw array into epochs shape
tmin, tmax = -.5, 2
epochs_audio = mne.Epochs(audio, ev, einfo, tmin, tmax,
                          baseline=(None, 0), preload=True)

# We'll decimate the data because we've got more datapoints than we need
epochs_audio.decimate(10)





