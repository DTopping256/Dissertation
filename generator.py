#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.utils import Sequence, to_categorical
import sys

# Allows me to import my modules
sys.path.append('./modules')
from audio_utils import *
from multiclass_labelling_utils import kit_combinations, kltls

kcs = kit_combinations()

global MULTILABEL
MULTILABEL = 0
global ONEHOT
ONEHOT = 1
global PROBLEM_TYPES
PROBLEM_TYPES = [MULTILABEL, ONEHOT]
    
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

class AudioGenerator(Sequence):
    def get_ys(self, labels):
        if (self.problem_type == MULTILABEL):
            return multilabelled_labels_to_ys(labels)
        else:
            return onehot_superclass_labels_to_ys(labels)
    
    def __init__(self, filenames, labels, data_type, batch_size, shuffle=False, problem_type=MULTILABEL):
        self.filenames, self.labels, self.batch_size, self.shuffle, self.problem_type = filenames, labels, batch_size, shuffle, problem_type
        # Initialised with ordered indexes
        self.indexes = np.arange(len(self.filenames))
        if (problem_type not in PROBLEM_TYPES):
            raise Exception("Invalid problem_type {}...\nNeeds to be one of: {}".format(problem_type, PROBLEM_TYPES))
        
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
            # Add data and label to batch arrays
            batch_x.append(data)
            batch_y.append(self.labels[i])
        output = np.array(batch_x).reshape(self.batch_size, 12000, 1), np.array(list(map(self.get_ys, batch_y)))
        return output
    
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            

