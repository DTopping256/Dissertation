#!/usr/bin/env python
# coding: utf-8

import statistics as stats
import sys

# Allows me to import my modules
sys.path.append('./modules')
from audio_utils import *

#### CROP CODE

# Load in raw (unedited) audio samples
samples = get_sample_data(data_type="raw", verbose=True)
no_of_samples = len(samples)

# Resample them to a lower sampling rate
resampled = [reduce_sample_rate(data=sample, new_rate=SETTINGS.cropping["down_sampling_rate"], verbose=True) for sample in samples]

# Get the onsets of each of the hits in the original samples, by analysing the resampled audio.
onsets = []
for i in range(no_of_samples):
    s = samples[i]
    rs = resampled[i]
    rs_ebn = energy_based_novelty(rs.data, SETTINGS.cropping["hanning_window_size"], True)
    rs_threshold = threshold_func(rs_ebn, int(SETTINGS.cropping["down_sampling_rate"]/5), 2*stats.stdev(rs_ebn), 5, True)
    rs_onsets = onset_detection(rs_ebn, rs_threshold, 1000, verbose=True)
    current_onsets = rs_onsets*int(s.rate/rs.rate)
    onsets.append(current_onsets)

# Using the onsets crop a length of audio from the original which captures a hit, save this crop.
for i in range(no_of_samples):
    s = samples[i]
    current_onsets = onsets[i]
    crop_and_save(data=s, start_points=current_onsets, length=0.25, overwrite_existing=True, verbose=True)