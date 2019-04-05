#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.utils import Sequence, to_categorical
import sys
from scipy import signal
from tensorflow.keras.utils import normalize

# Allows me to import my modules
sys.path.append('./modules')
from audio_utils import *
from multiclass_labelling_utils import kit_combinations, kltls

kcs = kit_combinations()

# Global variables to be given as arguments to the AudioGenerator initially:
global MULTI_LABEL
MULTI_LABEL = 0
global ONE_HOT
ONE_HOT = 1
global PROBLEM_TYPES
PROBLEM_TYPES = [MULTI_LABEL, ONE_HOT]
global TIME_SEQUENCE
TIME_SEQUENCE = 0
global LOG_SPECTROGRAM
LOG_SPECTROGRAM = 1
global LINEAR_SPECTROGRAM
LINEAR_SPECTROGRAM = 2
INPUT_TYPES = [TIME_SEQUENCE, LOG_SPECTROGRAM, LINEAR_SPECTROGRAM]
    
# Multi-label problem: labels -> output 
def multilabelled_labels_to_ys(labels):
    ys = np.zeros(len(kltls))
    for n in range(len(kltls)):
        kl, tl = kltls[n].split("-")
        for label_i in range(len(labels["hit_label"])):
            if (kl in labels["kit_label"][label_i] and tl in labels["tech_label"][label_i]):
                ys[n] = 1
    return ys

# Multi-label problem: output -> labels
def multilabelled_ys_to_labels(ys, threshold = 0.6):
    labels = {"hit_label": [], "kit_label": [], "tech_label": []}
    for n in range(len(kltls)):
        kl, tl = kltls[n].split("-")
        if (ys[n] >= threshold):
            hl = "beater" if kl == "bass_drum" else "stick"
            labels["hit_label"].append(hl)
            labels["kit_label"].append(kl)
            labels["tech_label"].append(tl)
    return labels

# Superclass problem: labels -> output
def onehot_superclass_labels_to_ys(labels):
    formatted_labels = []
    for i in range(len(labels["hit_label"])):
        kl, tl = labels["kit_label"][i], labels["tech_label"][i]
        formatted_labels.append("-".join([kl, tl]))
    label_i = kcs.index(set(formatted_labels))
    return to_categorical(label_i, len(kcs))

# Superclass problem: output -> labels
def onehot_superclass_ys_to_labels(ys, threshold = 0.6):
    labels = {"hit_label": [], "kit_label": [], "tech_label": []}
    highest_val = np.amax(ys)
    if (highest_val >= threshold):
        label_i = np.where(ys==highest_val)[0][0]
        kltls = list(kcs[label_i])
        kltls.sort()
        for kltl in kltls:
            kl, tl = kltl.split("-")
            hl = "beater" if kl == "bass_drum" else "stick"
            labels["hit_label"].append(hl)
            labels["kit_label"].append(kl)
            labels["tech_label"].append(tl)
    return labels

# FFT constants (tuned in spectrograms.ipynb):
FS = 48000 # sample rate of the audio files.
NFFT = 256 # size of the FFT's window function.
WINDOW_F = signal.windows.hann(NFFT) # creates an array of length NFFT, of points along a hanning curve; to be used as a window function for FFT.
NOVERLAP = 32 # size of the overlap between windows.

# Expecting an output of shape (129, 53) when input_type is LOG_SPECTROGRAM or LINEAR_SPECTROGRAM, for a audio file with 12000 samples.

# Getter for Sxx part of signal.spectrogram SciPy function, with our constants given some data as a parameter.
def get_spectrogram(data):
    return signal.spectrogram(data, fs=FS, window=WINDOW_F, nfft=NFFT, noverlap=NOVERLAP)[2]

# 10Log10 of a spectrogram with error correction on 0 terms caused by silence in translated audio.  
def clipped_log_spectrogram(data):
    Sxx = get_spectrogram(data)
    Sxx[Sxx == 0] = 1
    min_Sxx = np.amin(Sxx)
    Sxx[Sxx == 1] = min_Sxx
    return normalize(np.log10(Sxx))

class AudioGenerator(Sequence):
    def get_ys(self, labels):
        if (self.problem_type == MULTI_LABEL):
            return multilabelled_labels_to_ys(labels)
        else:
            return onehot_superclass_labels_to_ys(labels)
    
    def __init__(self, filenames, labels, data_type, batch_size, shuffle=False, problem_type=MULTI_LABEL, input_type=TIME_SEQUENCE):
        if (problem_type not in PROBLEM_TYPES):
            raise Exception("Invalid problem_type {}...\nNeeds to be one of: {}".format(problem_type, PROBLEM_TYPES))
        if (input_type not in INPUT_TYPES):
            raise Exception("Invalid input_type {}...\nNeeds to be one of: {}".format(problem_type, INPUT_TYPES))
            
        self.filenames, self.labels, self.batch_size = filenames, labels, batch_size
        self.shuffle, self.problem_type, self.input_type = shuffle, problem_type, input_type
        # Initialised with ordered indexes
        self.indexes = np.arange(len(self.filenames))
        
    def __len__(self):
        return int(np.floor(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        # Use ordering of self.indexes for dataset
        inds = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x, batch_y = [], []
        for i in inds:
            # Load data from .gz
            data = np.loadtxt(self.filenames[i])
            # Check and if necessary correct data length
            diff = len(data) - 12000
            if (diff < 0):
                data = np.pad(data, (0, abs(diff)), mode="constant")
            elif (diff > 0):
                data = data[:-diff]
            # Handles the shape of the chosen input data, adding a filter dimension.
            input_types = {
                TIME_SEQUENCE: lambda data: data.reshape(12000, 1),
                LOG_SPECTROGRAM: lambda data: clipped_log_spectrogram(data).reshape(129, 53, 1),
                LINEAR_SPECTROGRAM: lambda data: normalize(get_spectrogram(data).reshape(129, 53, 1))
            }
            # Add data and label to batch arrays
            batch_x.append(input_types[self.input_type](data))
            batch_y.append(self.labels[i])
        output = np.array(batch_x), np.array(list(map(self.get_ys, batch_y)))
        return output
    
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            

