import mne
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')

# Read in an MNE epochs file
data = mne.io.Raw('./data/audio_clean_raw.fif', add_eeg_ref=False)

# Load our times file, which is stored as a CSV of events
mtime = pd.read_csv('./data/time_info.csv', index_col=0)
mtime = mtime.query('t_type=="mid" and t_num > 0')

# We will create an "events" object by turning the start times into indices
# Then turning it into an array of shape (n_events, 3)
ev = mtime['start'] * data.info['sfreq']  # goes from seconds to index
ev = np.vstack([ev, np.zeros_like(ev), np.ones_like(ev)]).astype(int).T

# This is just a metadata dictionary.
# If we had multiple event types, we'd specify here.
einfo = dict(myevent=1)

# Now create an epochs object so we're time-locked to sound onset
tmin, tmax = -.5, 2
epochs = mne.Epochs(data, ev, einfo, tmin, tmax,
                    baseline=(None, 0), preload=True)

# Note it's the same shape as before
print(epochs._data.shape)

data_plt = epochs._data[0]
plt.plot(epochs.times, data_plt.T)

freqs = np.logspace(2, np.log10(5000), 128)
spec = mne.time_frequency.cwt_morlet(data_plt, epochs.info['sfreq'], freqs)

# Our output is now n_epochs x n_freqs x time
spec.shape

# And now we reveal the spectral content that was present in the sound
plt.pcolormesh(epochs.times, freqs, spec[0])

# Whoops, that looks pretty messy. Let's try taking the log...
f, ax = plt.subplots()
ax.pcolormesh(epochs.times, freqs, np.log(spec[0]))

# First we'll load some brain data
brain = mne.io.Raw('./data/ecog_clean_raw.fif', add_eeg_ref=False)

# We will create an "events" object by turning the start times into indices
ev = mtime['start'] * brain.info['sfreq']  # goes from seconds to index
ev = np.vstack([ev, np.zeros_like(ev), np.ones_like(ev)]).astype(int).T
einfo = dict(myevent=1)

# Now we have time-locked Epochs
tmin, tmax = -.5, 2
brain_epochs = mne.Epochs(brain, ev, einfo, tmin, tmax,
                          baseline=(None, 0), preload=True)

# Note it's the same shape as before
print(brain_epochs._data.shape)

# Pull a subset of epochs to speed things up
n_ep = 10
use_epochs = np.random.choice(range(len(brain_epochs)), replace=False, size=n_ep)
brain_epochs = brain_epochs[use_epochs]

# Multitaper
psd = []
for ep in brain_epochs._data:
    ipsd, freqs = mne.time_frequency.multitaper._psd_multitaper(
        ep, sfreq=brain_epochs.info['sfreq'])
    psd.append(ipsd)
psd = np.array(psd)
psd = pd.DataFrame(psd.mean(0), columns=freqs)
psd.index.name = 'elec'
psd['kind'] = 'mt'
psd.set_index('kind', append=True, inplace=True)

# Collect them
psd.columns.name = 'freq'

# Just as before, we'll apply the log and plot
psd.apply(np.log).groupby(level='kind').mean().T.plot(figsize=(15, 5))

# Here we'll define the range of frequencies we care about
freqs = np.logspace(1, np.log10(150), 20)

# This determines the length of the filter we use to extract spectral content
n_cycles = 5

freqs.shape

# Now we'll extract the TFR of our brain data
df_tfr = []
tfr, itc = mne.time_frequency.tfr_morlet(brain_epochs, freqs, n_cycles)
for i, elec in enumerate(tfr.data):
    ielec = pd.DataFrame(elec, index=freqs, columns=brain_epochs.times)
    ielec['elec'] = i
    ielec.index.name = 'freq'
    ielec.set_index(['elec'], inplace=True, append=True)
    df_tfr.append(ielec)
df_tfr = pd.concat(df_tfr, axis=0)

f, ax = plt.subplots()
plt_df = df_tfr.xs(20, level='elec')
y_axis = plt_df.index.values
# y_axis = np.arange(plt_df.shape[0])
ax.pcolormesh(plt_df.columns.values, y_axis, plt_df.values,
              cmap=plt.cm.RdBu_r)



