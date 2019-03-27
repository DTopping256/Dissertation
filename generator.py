#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.utils import Sequence
import sys

# Allows me to import my modules
sys.path.append('./modules')
from audio_utils import *

kltls = ['bass_drum-normal','hi_hat-normal',
  'hi_hat-open',
  'high_tom-normal',
  'ride-normal',
  'ride-bell',
  'crash-normal',
  'snare-normal',
  'low_tom-normal',
  'mid_tom-normal']

def labels_to_ys(labels):
    ys = np.zeros(len(kltls))
    for n in range(len(kltls)):
        kl, tl = kltls[n].split("-")
        for label_i in range(len(labels["hit_label"])):
            if (kl in labels["kit_label"][label_i] and tl in labels["tech_label"][label_i]):
                ys[n] = 1
    return ys

def ys_to_labels(ys, threshold = 0.6):
    labels = {"hit_label": [], "kit_label": [], "tech_label": []}
    for n in range(len(kltls)):
        kl, tl = kltls[n].split("-")
        if (ys[n] > threshold):
            hl = "beater" if kl == "bass_drum" else "stick"
            labels["hit_label"].append(hl)
            labels["kit_label"].append(kl)
            labels["tech_label"].append(tl)
    return labels

class AudioGenerator(Sequence):
    def get_ys(self, labels):
        return labels_to_ys(labels)
    
    def __init__(self, filenames, labels, data_type):
        self.filenames, self.labels, self.batch_size = filenames, labels, SETTINGS.data[data_type]["batch_size"]
        
    def __len__(self):
        return int(np.floor(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([np.loadtxt(file_name) for file_name in batch_x]).reshape(self.batch_size, 12000, 1), np.array(list(map(self.get_ys, batch_y)))

