# For loading and processing data
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.io import wavfile

# For making nifty widgets
from IPython.display import display
from ipywidgets import interact, widgets
get_ipython().magic('matplotlib inline')

# Here's a wavfile of me speaking
fs, data = wavfile.read('../data/science_is_awesome.wav')
times = np.arange(data.shape[0]) / float(fs)

f, ax = plt.subplots()
ax.plot(times, data)
_ = ax.set_title('The raw signal', fontsize=20)

f, axs = plt.subplots(2, 1, figsize=(10, 10))
axs[0].plot(times, data)
_ = axs[0].set_title('The raw signal', fontsize=20)
spec, freqs, spec_times, _ = axs[1].specgram(data, Fs=fs)
_ = axs[1].set_title('A spectrogram of the same thing', fontsize=20)
_ = plt.setp(axs, xlim=[times.min(), times.max()])
_ = plt.setp(axs[1], ylim=[freqs.min(), freqs.max()])

f, ax = plt.subplots(figsize=(10, 5))

nfft_widget = widgets.IntSlider(min=3, max=15, step=1, value=10)
n_overlap_widget = widgets.IntSlider(min=3, max=15, step=1, value=9)

def update_n_overlap(*args):
    n_overlap_widget.value = np.clip(n_overlap_widget.value, None, nfft_widget.value-1)
nfft_widget.observe(update_n_overlap, 'value')
n_overlap_widget.observe(update_n_overlap, 'value')

def func(n_fft, n_overlap):
    spec, freqs, spec_times, _ = ax.specgram(data, Fs=fs,
                                        NFFT=2**n_fft, noverlap=2**n_overlap,
                                        animated=True)
    ax.set(xlim=[spec_times.min(), spec_times.max()],
           ylim=[freqs.min(), freqs.max()])
    plt.close(f)
    display(f)
w = interact(func, n_fft=nfft_widget,
             n_overlap=n_overlap_widget)

# Note that now the sliders won't update until you release the mouse to save time.
f, ax = plt.subplots(figsize=(10, 5))
def func(n_cycles, n_freqs):
    plt.close(f)
    freqs = np.logspace(np.log10(100), np.log10(20000), n_freqs)
    amps = mne.time_frequency.cwt_morlet(data[np.newaxis, :], fs,
                                         freqs, n_cycles=n_cycles)
    amps = np.log(np.abs(amps))[0]
    ax.imshow(amps, animated=True, origin='lower', aspect='auto')
    display(f)
    
n_cycles_widget = widgets.IntSlider(min=5, max=50, step=1, value=3, continuous_update=False)
n_freqs_widget = widgets.IntSlider(min=10, max=150, step=10, value=50, continuous_update=False)
w = interact(func, n_cycles=n_cycles_widget, n_freqs=n_freqs_widget)



