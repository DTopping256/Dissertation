#!/usr/bin/env python
# coding: utf-8

import statistics as stats
import sys

# Allows me to import my modules
sys.path.append('./modules')
from audio_utils import *

#### CROP CODE

i = 0
while True:
    # Load in raw (unedited) audio samples
    samples = get_data(data_type="raw", batch_index=i, verbose=True)
    if samples is False:
        break
    i += 1

    no_of_samples = len(samples)
    data_info = SETTINGS.data["cropped"]

    # Resample them to a lower sampling rate
    resampled = [reduce_sample_rate(data=sample, new_rate=data_info["down_sampling_rate"], verbose=True) for sample in samples]

    # Get the onsets of each of the hits in the original samples, by analysing the resampled audio.
    onsets = []
    for i in range(no_of_samples):
        s = samples[i]
        rs = resampled[i]
        rs_ebn = energy_based_novelty(rs.data, data_info["hanning_window_size"], verbose=True)
        rs_threshold = threshold_func(rs_ebn, int(data_info["down_sampling_rate"]/5), 2*stats.stdev(rs_ebn), 5, verbose=True)
        rs_onsets = onset_detection(rs_ebn, rs_threshold, data_info["onset_trim"], verbose=True)
        current_onsets = rs_onsets*int(s.rate/rs.rate)
        onsets.append(current_onsets)

    # Using the onsets crop a length of audio from the original which captures a hit, save this crop.
    cropped_samples = []
    for i in range(no_of_samples):
        s = samples[i]
        current_onsets = onsets[i]
        for onset in current_onsets:
            # Make a crop in the np array of samples
            start = onset
            end = int(start+data_info["length"]*s.rate)
            cropped_samples.append(Sample_data(labels=s.labels, augmentations=s.augmentations, rate=s.rate, data=s.data[start:end]))  
    
    save_data(data_type="cropped", data_set=cropped_samples, verbose=True)