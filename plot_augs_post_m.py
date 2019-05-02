#!/usr/bin/env python
# coding: utf-8

import sys
import collections
import numpy as np
from scipy.signal import savgol_filter, convolve, normalize
import matplotlib.pyplot as plt
import librosa
import time

# Allows me to import my modules
sys.path.append('./modules')
from audio_utils import *
import data_augmentation as data_aug

file_i = 0

samples = read_data(data_type="cropped", file_index=file_i, batch_size=4, verbose=True)

augmented_samples = data_aug.post_multiclass_augmentation(samples)

plt.plot(augmented_samples[3]["add_noise"][-1], color="grey")
plt.plot(samples[3].data, color="blue")
plt.title("Add noise augmentation")
plt.xlabel("Time (sample number)")
plt.ylabel("Amplitude (arb. unit)")
plt.legend(["Augmented", "Not augmented"])
plt.show()

plt.plot(samples[0].data, color="blue")
plt.plot(augmented_samples[0]["reduce_noise"][-1], color="grey")
plt.title("Reduce noise augmentation")
plt.xlabel("Time (sample number)")
plt.ylabel("Amplitude (arb. unit)")
plt.legend(["Not augmented", "Augmented"])
plt.show()