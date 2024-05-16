from __future__ import print_function

import librosa
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

y, sr = librosa.load('audio/Karissa_Hobbs_-_09_-_Lets_Go_Fishin.mp3', sr=44100, duration=2, offset=35)

# Parameters from the paper:

n_fft = 1024
hop_length = int(librosa.time_to_samples(1./200, sr=sr))
lag = 2
n_mels = 138
fmin = 27.5
fmax = 16000.
max_size = 3

hop_length

S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax, n_mels=n_mels)

plt.figure(figsize=(6, 4))
librosa.display.specshow(librosa.logamplitude(S, ref_power=np.max),
                         y_axis='mel', x_axis='time', sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)
plt.tight_layout()

odf_default = librosa.onset.onset_strength(y=y, sr=sr)
onset_default = librosa.onset.onset_detect(y=y, sr=sr)

odf_sf = librosa.onset.onset_strength(S=librosa.logamplitude(S), sr=sr, hop_length=hop_length,
                                      lag=lag, max_size=max_size)

onset_sf = librosa.onset.onset_detect(onset_envelope=odf_sf, sr=sr, hop_length=hop_length)

plt.figure(figsize=(6, 6))

plt.subplot(2,1,2)
librosa.display.specshow(librosa.logamplitude(S, top_db=50, ref_power=np.max),
                         y_axis='mel', x_axis='time', sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax,
                         n_xticks=9)

plt.subplot(4,1,1)
plt.plot(odf_default, label='Spectral flux')
plt.vlines(onset_default, 0, odf_default.max(), color='r', label='Onsets')
plt.yticks([])
plt.xticks([])
plt.axis('tight')
plt.legend()


plt.subplot(4,1,2)
plt.plot(odf_sf, color='g', label='Superflux')
plt.vlines(onset_sf, 0, odf_sf.max(), color='r', label='Onsets')
plt.xticks([])
plt.yticks([])
plt.legend()
plt.axis('tight')

plt.tight_layout()

