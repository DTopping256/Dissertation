#!/usr/bin/env python
# coding: utf-8

import json
import math
import numpy as np
import os
from scipy import signal
from scipy.io import wavfile as wav
import statistics as stats
import sys
import time

# Returns settings dictionary from the settings.json file.
def get_settings(verbose=False):
    if verbose:
        print("Attempting to read settings file...")
    try:
        with open("settings.json", "r") as settings_file:
            if verbose:
                print("\tRead successfully!")
            return json.load(settings_file)
    except:
        print("\tError reading the settings file:\n\t\t" if verbose else "Error: ", sys.exc_info())
        return None

# A wrapper for the settings
class Settings:
    def __new__(cls, *args, **kwargs):
        verbose = args[0] if len(args) > 0 else kwargs["verbose"] if len(kwargs) > 0 else False
        s = get_settings(verbose)
        if (s is not None):
            cls.data_paths = {k: v for k, v in s["data_paths"].items()}
            cls.cropping = {k: v for k, v in s["cropping"].items()}
            cls.dir_structure = s["dir_structure"]
            cls.hit_labels = s["hit_labels"]
            cls.kit_labels = s["kit_labels"]
            cls.tech_labels = s["tech_labels"]
            instance = super(Settings, cls).__new__(cls)
            return instance
        else:
            return None
    def __init__(self, verbose=False):
        pass

SETTINGS = Settings(verbose=True)

# Wrapper for audio: labels, audio data (samples) and the rate (sample rate) 
class Sample_data:
    def __init__(self, labels=None, rate=None, data=None):
        self.labels = labels
        self.data = np.array(data)
        self.rate = rate

# Returns either a .wav files' (samples, sample_rate) or (None, None) if an error occurred.
def silent_wav_read(filepath, verbose=False):
    try:
        return wav.read(filepath)
    except:
        if verbose: 
            print("\tError reading wav file: " + filepath + "\n\t", *sys.exc_info())
        return (None, None)

# Gets a specified type of audio data from one of the data_paths in settings.json. Returns this data as an array of sample_data instances. 
def get_sample_data(data_type="raw", verbose=False):
    if verbose and labelled:
        print("Attempting to read " + data_type + " data and labels...")
    if (SETTINGS is None):
        print("\tRead failed, since settings not found.")
        return False
    path = os.path.join(os.getcwd(), SETTINGS.data_paths[data_type])
    wavs_found = 0
    wavs_read = 0
    output = []
    # Search the path for files and subdirectories
    for path, subdirs, files in os.walk(path):
        # Filter non .wav files
        files = list(filter(lambda filename: filename[-4:] == ".wav", files))
        file_count = len(files)
        wavs_found += file_count
        if (file_count > 0):
            dir_structure = SETTINGS.dir_structure
            label_dirs = path.split(os.sep)[-3:]
            # Assign labels from directory structure to a dictionary.
            labels = {dir_structure[l]: label_dirs[l] for l in range(len(dir_structure))}
            # Returns an array of sample_data class instances which have audio data.
            data = list(filter(lambda sample: sample.data is not None, [Sample_data(labels, *silent_wav_read(os.path.join(path, file))) for file in files]))
            wavs_read += len(data)
            output.extend(data)
    if verbose:
        print("\tRead {}/{} audio files".format(wavs_read, wavs_found))
    return output

# Gets audio data from a specific directory, not going into subdirectories. Returns this data as an array of sample_data instances. 
def get_sample_data_from(directory=None, verbose=False):
    if verbose and directory is not None:
        print("Attempting to read data from " + directory + " ...")
    else:
        print("\tMust provide directory arg")
        return False
    path = os.path.join(os.getcwd(), directory)
    wavs_found = 0
    wavs_read = 0
    # Search the path for files and subdirectories
    content = os.listdir(path)
    # Filter non .wav files
    files = list(filter(lambda filename: filename[-4:] == ".wav", content))
    file_count = len(files)
    wavs_found += file_count
    if (file_count > 0):
        data = [Sample_data({}, *silent_wav_read(os.path.join(path, filepath), verbose=verbose)) for filepath in files]
        #data = list(filter(lambda sample: sample.data is not None, data))
        if verbose:
            print("\tRead {} audio files".format(wavs_found))
        return data
    return False

# With a given sample_data instance, resamples it into approximately the new sample rate; returning it as a new sample_data instance.
def reduce_sample_rate(data, new_rate, verbose=False):
    start_time = time.time()
    no_of_samples = len(data.data)
    t = no_of_samples/data.rate
    if verbose:
        print("Resampling {}s of data: from {} samples/s to ~{} samples/s".format(str(round(t, 3)), data.rate, new_rate))
    # Calculating the new number of samples
    num = int(new_rate/data.rate*no_of_samples)
    resampled_data = signal.resample(data.data, num)
    new_rate = round(num/t, 4)
    end_time = time.time()
    if verbose:
        print("\tDone to {} samples/s! (Execution time: {}s)".format(new_rate, round(end_time-start_time, 2)))
    return Sample_data(data.labels, new_rate, resampled_data)


## Onset detection stuff
# Converting amplitude into local energy
def local_energy_func(samples, N):
    s_len = len(samples)
    # N length, Hanning window function
    w = signal.hann(N)
    energy_signal = []
    # Apply the "local energy" formula on the samples.
    for s in range(s_len):
        energy = 0
        for n in range(N):
            i = int(s+n-N/2)
            w_val = w[n]
            energy += (samples[i]*w_val)**2 if i >= 0 and i < s_len else 0
        energy_signal.append(energy)
    return np.array(energy_signal)

# Converting amplitude into energy based novelty (a detection function)
def energy_based_novelty(samples, N, verbose=False):
    if verbose:
        print("Calculating ebn of samples...")
    start_time = time.time()
    # Get the local energy of the samples
    sample_local_energy = local_energy_func(samples, N)
    # Get the energy based novelty using the log(Δenergy) 
    energy_novelty = []
    prev_energy = sample_local_energy[0]
    for energy in sample_local_energy:
        energy_novelty.append(abs(math.log(energy/prev_energy)))
        prev_energy = energy
    end_time = time.time()
    if verbose:
        print("\tDone. (Execution time: {}s)".format(round(end_time-start_time, 2)))
    return np.array(energy_novelty)

# Finding the variable threshold of the detection function (ebn) using the median and the c1 & c2 constants.
def threshold_func(ebn, N, c1, c2, verbose=False):
    if verbose:
        print("Creating a threshold function for enb function...")
    start_time = time.time()
    ebn_len = len(ebn)
    half_N = int(N/2)
    padding = np.array([0.0 for i in range(half_N)])
    padded_ebn = np.concatenate((padding, ebn, padding))
    detection_thresholds = []
    for dv in range(half_N, ebn_len+half_N):
        ebn_seg = padded_ebn[dv-half_N:dv+half_N]
        median = np.median(np.abs(ebn_seg))
        detection_thresholds.append(c1+c2*median)
    end_time = time.time()
    if verbose:
        print("\tDone. (Execution time: {}s)".format(end_time-start_time, 2))
    return np.array(detection_thresholds)

# Finding the indexes of onsets (where the ebn function is initially greater than the threshold function) when the functions are trimmed by an amount of samples either end of the audio.
def onset_detection(ebn, threshold, trim=1000, verbose=False):
    if verbose:
        print("Finding onsets (peak peaking the detection function)...")
    start_time = time.time()
    onsets = []
    prev_grad = None
    for i in range(trim, int(len(ebn)-trim*2)):
        g = ebn[i+1]-ebn[i]
        above_threshold = ebn[i] > threshold[i]
        # Find the peaks in ebn above the threshold function
        if (above_threshold and g < 0 and prev_grad > 0):
            onsets.append(i)
        prev_grad = g
    end_time = time.time()
    if verbose:
        print("\t{} onsets found. (Execution time: {}s)".format(len(onsets), round(end_time-start_time, 2)))
    return np.array(onsets)

def crop_and_save(data, start_points, length, overwrite_existing=False, verbose=False):
    if verbose:
        print("Attempting to crop and save audio...")
    if (SETTINGS is None):
        if verbose:
            print("\tRead failed, since settings not found.")
        return False
    uid=None
    success = 0
    total = len(start_points)
    if verbose:
        print("\tOverwriting [{}]: {}".format(total, filepath) if overwrite_existing else "\tCreating [{}]: {}".format(total, filepath))
    for i in range(total):
        # Make a crop in the np array of samples
        start = start_points[i]
        end = int(start+length*data.rate)
        crop = np.array(data.data[start:end])  
        
        # Find the output file, if file or directory structure not present attempt to create it.
        subdirs = [data.labels[label] for label in SETTINGS.dir_structure]
        path = os.path.join(os.getcwd(), SETTINGS.data_paths["cropped"], *subdirs)
        if (not os.path.exists(path)):
            try:
                os.makedirs(path)
            except OSError:
                if verbose:
                    print("\t\tCouldn't make new folder: ", path)
                return False
        file_uid = str(i) if uid is None else str(uid)
        possible_filename = file_uid + ".wav"
        possible_filepath = os.path.join(path, possible_filename)
        while(os.path.isfile(possible_filepath) and not overwrite_existing):
            uid = i+1 if uid is None else uid+1
            possible_filepath = os.path.join(path, str(uid) + ".wav")
        filepath = possible_filepath
        # Write crop to output filepath
        try:
            with open(filepath, 'w' if overwrite_existing else 'w+') as f:
                wav.write(filepath, rate=int(data.rate), data=crop)
        except:
            print("\t\tUnexpected error:\n\t\t", sys.exc_info())
            continue
        success += 1
    if verbose:
        print("\tSuccesfully saved: {}/{} crop segments".format(success, total))
    return success == total

def save(path, samples, rate, verbose=False):
    # Find the output file, if file or directory structure not present attempt to create it.
    currentfiles=[]
    if verbose:
        print("Saving audio")
    if (not os.path.exists(path)):
        try:
            os.makedirs(path)
        except OSError:
            if verbose:
                print("\tCouldn't make new folder: ", path)
            return False
    else:
        currentfiles = os.listdir(path)
        # Filter non .wav files
        currentfiles = list(map(lambda filename: int(filename[:-4]), filter(lambda filename: filename[-4:] == ".wav", currentfiles)))
        currentfiles.sort(reverse=True)
    possible_filename = "0.wav"
    if len(currentfiles) > 0:
        possible_filename = str(currentfiles[0]+1)+".wav"
    filepath = os.path.join(path, possible_filename)
    # Write crop to output filepath
    try:
        with open(filepath, 'w+') as f:
            wav.write(filepath, rate=int(rate), data=samples)
    except:
        print("\tUnexpected error:\n\t\t", sys.exc_info())
        return False
    return True