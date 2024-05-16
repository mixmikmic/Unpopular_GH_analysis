from __future__ import print_function

# Note the _ prefix: we use this to alias the unmodified module
import librosa as _librosa

from presets import Preset

# The Preset object wraps a module (and its submodules) with a dictionary interface
librosa = Preset(_librosa)

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# We can now change some default settings via the dictionary

# Use 44.1KHz as the default sampling rate
librosa['sr'] = 44100

# Change the FFT parameters
librosa['n_fft'] = 4096
librosa['hop_length'] = librosa['n_fft'] // 4

y, sr = librosa.load(librosa.util.example_audio_file())

print(sr)

S = librosa.stft(y)

print(S.shape)

print(librosa.get_duration(S=S), librosa.get_duration(y=y, sr=sr))

plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max), x_axis='time', y_axis='log')
plt.tight_layout();

# Presets can be explicitly overridden just like any other default value:

S = librosa.stft(y, hop_length=2048)

print(S.shape)

plt.figure(figsize=(12, 4))
librosa.display.specshow(librosa.logamplitude(S**2, ref_power=np.max), x_axis='time', hop_length=2048, y_axis='log')
plt.tight_layout();

