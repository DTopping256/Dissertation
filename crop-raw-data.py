#!/usr/bin/env python
# coding: utf-8

import statistics as stats
import sys
import gc

# Allows me to import my modules
sys.path.append('./modules')
from audio_utils import *

#### CROP CODE
raw_settings = SETTINGS.data["raw"]
crop_settings = SETTINGS.data["cropped"]

file_i = 0
while True:
    # Load in raw (unedited) audio samples
    samples = read_data(data_type="raw", file_index=file_i, batch_size=raw_settings["batch_size"], verbose=True)
    if samples is False:
        break
    file_i += 1

    no_of_samples = len(samples)

    # Resample them to lower sample rates
    resampled = [reduce_sample_rate(data=sample, new_rate=crop_settings["down_sampling_rate"], verbose=True) for sample in samples]
    save_resamples = [reduce_sample_rate(data=sample, new_rate=crop_settings["save_sampling_rate"], verbose=True) for sample in samples]
    
    # Free up (original) samples data
    del samples
    gc.collect()

    # Get the onsets of each of the hits in the original samples, by analysing the resampled audio.
    onsets = []
    for i in range(no_of_samples):
        s = save_resamples[i]
        rs = resampled[i]
        rs_ebn = energy_based_novelty(rs.data, crop_settings["hanning_window_size"], verbose=True)
        rs_threshold = threshold_func(rs_ebn, int(crop_settings["down_sampling_rate"]/5), 2*stats.stdev(rs_ebn.tolist()), 5, verbose=True)
        rs_onsets = onset_detection(rs_ebn, rs_threshold, crop_settings["onset_trim"], verbose=True)
        current_onsets = rs_onsets*int(s.rate/rs.rate)
        onsets.append(current_onsets)

    # Using the onsets crop a length of audio from the original which captures a hit, save this crop.
    cropped_samples = []
    dropped, total_onset_no = 0, 0
    for i in range(no_of_samples):
        s = save_resamples[i]
        current_onsets = onsets[i]
        for onset in current_onsets:
            total_onset_no += 1
            # Make a crop in the np array of samples
            start = onset
            end = int(start+crop_settings["length"]*s.rate)
            crop = np.float32(s.data[start:end])
            if (len(crop) > 0 and np.amax(crop) > crop_settings["noise_threshold"]):
                cropped_samples.append(Sample_data(labels={k:v for k,v in s.labels.items()}, augmentations={k:v for k,v in s.augmentations.items()}, rate=int(s.rate), data=crop))  
            else:
                dropped += 1

    if(dropped > 0):
        print("{}/{} onsets dropped. (didn't meet noise threshold)".format(dropped, total_onset_no))

    # Free up onsets, resampled and save_samples data
    del onsets
    del resampled
    del save_resamples
    gc.collect()

    # Save cropped samples
    save_data(data_type="cropped", data_set=cropped_samples, verbose=True)

    # Free up cropped samples
    del cropped_samples
    gc.collect()