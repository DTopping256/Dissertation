#!/usr/bin/env python
# coding: utf-8

import sys
import collections
import numpy as np
from scipy.signal import savgol_filter
import librosa
import time
from itertools import permutations

# Allows me to import my modules
sys.path.append('./modules')
from audio_utils import *

# Get all wav files in cropped directory.
data = get_sample_data(data_type="cropped")

def white_noise(amount, mu, sigma_squared):
    return sigma_squared*np.random.randn(amount)+mu

def data_aug_white_noise(samples, noise_amp):
    return samples + white_noise(len(samples), 0, 0.175*(noise_amp+1))

def data_aug_reduce_noise(samples, window_i):
    return savgol_filter(samples, 100*(i+1)+1, 2)

def data_aug_amplitude(samples, target_amplitude):
    highest_amplitude = np.amax(samples)
    multiplier = target_amplitude/highest_amplitude
    return samples*multiplier

def data_aug_pitch_shift(samples, pitch_diff, sample_rate):
    if (type(samples) is not np.float32):
        samples = np.float32(samples)
    return librosa.effects.pitch_shift(y=samples, sr=sample_rate, n_steps=pitch_diff)

augmentations = {"amplitude": data_aug_amplitude, "pitch": data_aug_pitch_shift, "add_noise": data_aug_white_noise, "reduce_noise": data_aug_reduce_noise}

# Permutations of 2 or more length of augmentation operations.
aug_stack =  [[c1, c2] for c1, c2 in permutations(augmentations.keys(), 2)] + [[c1, c2, c3] for c1, c2, c3 in permutations(augmentations.keys(), 3)] + [[c1, c2, c3, c4] for c1, c2, c3, c4 in permutations(augmentations.keys(), 4)]

# Add single augmentations to the set 
aug_stack = aug_stack + [[aug] for aug in augmentations.keys()]
aug_data = collections.OrderedDict([("-".join(stack), []) for stack in aug_stack])
aug_keys = list(aug_data.keys())
stacks = len(aug_keys)

# Apply augmentation stacks to the data and save these augmentations.
augmented_data = []
data_size = len(data)
for j in range(int(data_size/4)):
    
    print("Processing augmented data - {}/{}".format(j, data_size))
    # Process augmentation stacks of data in batches
    processing_start_time = time.time()
    for d in data[j*4:(j+1)*4]:
        aug_data = collections.OrderedDict([("-".join(stack), []) for stack in aug_stack])
        for s in range(stacks):
            stack_key = aug_keys[s]
            aug_samples = d.data
            for aug in aug_stack[s]:
                for i in range(10):
                    args = []
                    var = i+1
                    if aug == "amplitude":
                        var = np.amax(d.data)
                        var = var - (5-i)*5 if i < 5 else var + (i-4)*5
                    if aug == "pitch":
                        args = [d.rate]
                        var = i - 5 if i < 5 else i - 4
                    aug_samples = augmentations[aug](aug_samples, var, *args)
            aug_data[stack_key].append(aug_samples)
        augmented_data.append((aug_data, d.labels, d.rate))
    processing_end_time = time.time()
    
    print("\tExecuted in: {}\nSaving augmented data - {}/{}".format(round(processing_end_time-processing_start_time, 2), j, data_size))
    # Save augmented data as seperate wav files in batches
    for aug_data, labels, rate in augmented_data[j*4:(j+1)*4]:
        subdirs = [labels[label] for label in SETTINGS.dir_structure]
        for aug in aug_keys:
            path = os.path.join(SETTINGS.data_paths["preprocessed"], aug, *subdirs)
            for samples in aug_data[aug]:
                save(path, np.array(samples, dtype=np.int16), rate)
    saving_end_time = time.time()
    print("\tExecuted in:", round(saving_end_time-processing_end_time, 2))

# Save augmented data in one place as a zipped file.
np.savetxt(fname=os.path.join(os.getcwd(), "preprocessed_data", "data.gz"), X=np.array(augmented_data), fmt="%s")