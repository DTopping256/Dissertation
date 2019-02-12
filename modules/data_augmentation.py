#!/usr/bin/env python
# coding: utf-8

import sys
import collections
import numpy as np
from scipy.signal import savgol_filter
import librosa
import time
from itertools import combinations 
from functools import reduce

# Allows me to import my modules
sys.path.append('./modules')
from audio_utils import *

def diff_m(diff, max_diff):
    mid_point = math.ceil(max_diff/2)
    if (diff > mid_point):
        return mid_point-diff
    else:
        return diff

def white_noise(amount, mu, sigma_squared):
    return sigma_squared*np.random.randn(amount)+mu

# Augmentation functions
def data_aug_white_noise(samples, diff, max_diff, **kwargs):
    highest_amplitude = np.amax(samples)
    noise_amplitude = 0.1*highest_amplitude*diff/max_diff
    return samples + white_noise(len(samples), 0, noise_amp)

def data_aug_reduce_noise(samples, diff, max_diff, **kwargs):
    window_len = 2*int(150*diff/max_diff)+1
    return savgol_filter(samples, window_len, 2)

def data_aug_amplitude(samples, diff, max_diff, **kwargs):
    highest_amplitude = np.amax(samples)
    target_amplitude = 0.8*diff/max_diff
    multiplier = target_amplitude/highest_amplitude
    return samples*multiplier

def data_aug_pitch_shift(samples, diff, max_diff, **kwargs):
    pitch_diff = diff_m(diff, max_diff)
    return librosa.effects.pitch_shift(y=samples, sr=kwargs["sample_rate"], n_steps=pitch_diff)

def data_aug_translate(samples, diff, max_diff, **kwargs):
    translate_diff = int(0.05*kwargs["sample_rate"]*diff/max_diff)
    rolled_samples = np.roll(samples, translate_diff)
    rolled_samples[:translate_diff] = 0
    return rolled_samples

PRE_MULTI_AUGS = {"amplitude": data_aug_amplitude, "pitch": data_aug_pitch_shift, "translate": data_aug_translate}
POST_MULTI_AUGS = {"add_noise": data_aug_white_noise, "reduce_noise": data_aug_reduce_noise}

# Combinations of all used augmentation operations (alphabetically ordered).
def get_aug_stacks(augmentations):
    orderedAugKeys = list(augmentations.keys())
    orderedAugKeys.sort()
    aug_stacks = [[a] for a in orderedAugKeys]
    for i in range(2, len(orderedAugKeys)):
        aug_stacks.extend([[*a] for a in combinations(orderedAugKeys, i)])
    aug_stacks.append(orderedAugKeys)
    return aug_stacks

# Performs all of the pre-multiclass augmentations in every combination a certain amount of times with varied strength to the data-set. 
def pre_multiclass_augmentation(data_set):
    aug_info = SETTINGS.data["pre_multiclass_augmented"]
    augmentations = PRE_MULTI_AUGS
    aug_stacks = get_aug_stacks(augmentations)

    # Create new augmentation data ordered dictionary
    aug_data = collections.OrderedDict([("-".join(stack), []) for stack in aug_stacks])
    aug_keys = list(aug_data.keys())
    stacks = len(aug_keys)

    all_augmented_data = []
    data_size = len(data_set)
    for data_i in range(data_size):
        data = data_set[data_i].data
        sample_rate = data_set[data_i].rate
        print("Processing augmented data - {}/{} ({}%)".format(data_i, data_size, round(data_i/data_size*100, 2)))
        # Per stack, augment the data with the augmentations in the stack
        processing_start_time = time.time()
        augmented_data = []
        for stack in range(stacks):
            aug_stack = aug_stacks[stack]
            stack_len = len(aug_stack)
            stack_is = [s_i for s_i in range(stack_len)]
            stack_key = aug_keys[stack]
            aug_stack_samples = []
            # Combinations of all augmentation diffs
            for diff in [[i//aug_info["amount"]**j%aug_info["amount"]+1 for j in range(stack_len)] for i in range(aug_info["amount"]**stack_len)]:
                aug_stack_samples.append(reduce(lambda d, s_i: augmentations[aug_stack[s_i]](d, diff=diff[s_i], max_diff=aug_info["amount"], sample_rate=sample_rate), stack_is, data))
            augmented_data.append((stack_key, aug_stack_samples))
        all_augmented_data.append(collections.OrderedDict(augmented_data))
    processing_end_time = time.time()
    print("Time taken:", round(processing_end_time-processing_start_time, 2), "\n")
    return all_augmented_data

