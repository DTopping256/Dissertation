#!/usr/bin/env python
# coding: utf-8

# In[29]:


import os
import sys
import numpy as np
import gzip
import itertools
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import soundfile as sf
import statistics as stats
import struct

# Allows me to import my modules
sys.path.append('./modules')
from audio_utils import *


beatdata = sf.read(os.path.join("audio_data", "raw_data", "drum_beats", "8_beat_sticks.wav"))

no_save_samples = int(48000/beatdata[1]*len(beatdata[0]))
resampled_save_beatdata = signal.resample(beatdata[0], no_save_samples)
no_onset_samples = int(3000/beatdata[1]*len(beatdata[0]))
resampled_onset_beatdata = signal.resample(beatdata[0], no_onset_samples)
ebn = energy_based_novelty(resampled_onset_beatdata, 100)
tf = threshold_func(ebn, 100, stats.stdev(ebn.tolist()), 2)
onsets = onset_detection(ebn, tf, 3000)

first_onset_start = onsets[2]*int(48000/3000)
first_onset_finish = first_onset_start+12000
plt.plot(resampled_save_beatdata[first_onset_start:first_onset_finish])
plt.title("Recorded bass drum with hi-hat")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude (arb. unit)")
plt.show()

file_data = get_file_classes("multiclassed_no_augs")
file_data

hh_bd_data = list(filter(lambda fd: set(fd["labels"]["kit_label"]) == set(["bass_drum", "hi_hat"]) and set(fd["labels"]["tech_label"]) == set(["normal", "normal"]), file_data))
hh_bd = np.loadtxt(hh_bd_data[0]["filepath"])

plt.plot(hh_bd)
plt.title("Superimposed bass drum and hi-hat")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude (arb. unit)")
plt.show()