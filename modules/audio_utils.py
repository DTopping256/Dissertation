#!/usr/bin/env python
# coding: utf-8

import json
import math
from functools import reduce
from itertools import combinations
import numpy as np
import os
from scipy import signal
import soundfile as sf
import statistics as stats
import sys
import time
import gzip

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
            cls.data = s["data"]
            cls.label = s["label"]
            cls.hierarchy = s["label-hierarchy"]
            instance = super(Settings, cls).__new__(cls)
            return instance
        else:
            return None
    def __init__(self, verbose=False):
        pass

SETTINGS = Settings(verbose=True)
AUG_KEYS = ["pre_augs", "post_augs"]
PRE_AUGS = SETTINGS.data["pre_multiclass_augmented"]["augmentation"]
POST_AUGS = SETTINGS.data["post_multiclass_augmented"]["augmentation"]
LABELS = list(SETTINGS.label.keys())

# Wrapper for audio: labels, audio data (samples) and the rate (sample rate) 
class Sample_data:
    def __init__(self, labels=None, augmentations=None, rate=None, data=None):
        if not (type(data) is np.ndarray and data.dtype is np.dtype(np.float32)):
            raise Exception("Data isn't the correct type. (np.float32)")
        self.labels = labels
        self.augmentations = augmentations
        self.data = data
        self.rate = rate

def dir_structure_check(dir_structure_key, item):
    classes = []
    if dir_structure_key in LABELS:
        classes = SETTINGS.label[dir_structure_key]
    elif "pre_augs" == dir_structure_key:
        classes = PRE_AUGS
    elif "post_augs" == dir_structure_key:
        classes = POST_AUGS
    return item in classes

## Reading and writing audio data utilities (with labels and augmentations as metadata)
'''
Read a specified type of audio data from one of the data_paths in settings.json.
Returns this data as an array of sample_data instances, which include label and augmentation information. 
Can filter the data it retrieves by label and augmentation if included in kwargs. 
'''
def read_data(data_type, file_index=0, batch_size=None, verbose=False, **kwargs):
    if verbose:
        print("Attempting to read", str(batch_size), data_type, "data and labels (from index", str(file_index), ")...")
    if (SETTINGS is None):
        print("\tRead failed, since settings not found.")
        return False
    if (data_type not in SETTINGS.data.keys()):
        print("\t{} is not a valid data_type.")
        return False
    if (file_index is False):
        print("\tNo batch index provided.")
        return False
    data_info = SETTINGS.data[data_type]
    including = {k: list(filter(lambda item: dir_structure_check(k, item), v)) for k, v in kwargs.items() if k in data_info["dir_structure"]}
    if len(including.keys()) > 0:
        print("\tOnly" + str(including))
    root_path = os.path.join(os.getcwd(), data_info["path"])
    files_found, files_attempted, files_read = 0, 0, 0
    file_ext = "."+data_info["file_type"]
    batch_size = data_info["batch_size"] if batch_size is None else batch_size

    # Search the path for all files and sub-directories of the type
    file_data = []
    for path, subdirs, files in os.walk(root_path):
        # Filter non .wav files
        files = list(filter(lambda filename: filename[-len(file_ext):] == file_ext, files))
        file_count = len(files)
        if (file_count > 0):
            dir_structure_len = len(data_info["dir_structure"])
            dirs = path.split(os.sep)[-dir_structure_len:]
            # Assign labels from directory structure to a dictionary.
            classes = {data_info["dir_structure"][l]: dirs[l].split("-") for l in range(dir_structure_len)}
            labels = {k: v for k, v in classes.items() if k in LABELS}
            augmentations = {k: v for k, v in classes.items() if k in PRE_AUGS+POST_AUGS}
            if (len(including.keys()) > 0):
                skip = False
                for k, v in including.items():
                    if set(v) not in [set([*n]) for n in combinations(labels[k], len(v))]:
                        skip = True
                        break
                if (skip):
                    continue
            files_found += file_count
            file_data.extend([{"filepath": os.path.join(path, f), "labels": labels, "augmentations": augmentations} for f in files])

    # Returns an array of sample_data class instances which have audio data. For the selected batch in the found files
    fd_amount = len(file_data)
    file_index *= batch_size
    if (file_index >= fd_amount):
        print("File index too high, file doesn't exist.")
        return False
    output = []
    for d in file_data[file_index: file_index + batch_size]:
        sample_data, sample_rate = None, None
        filepath = d["filepath"]
        try:
            files_attempted += 1
            if file_ext == ".wav":
                with open(filepath, "r") as fh:
                    sample_data, sample_rate = sf.read(file=filepath, dtype="float32")
            elif file_ext == ".gz":
                with gzip.open(filepath, "r") as fh:
                    line = fh.readline()
                    sample_rate = int(float(str(line)[4:-5]))
                sample_data = np.loadtxt(fname=filepath, dtype=np.float32)
        except:
            if verbose: 
                print("\tError reading {} file: {}\n\t{}".format(file_ext, filepath, *sys.exc_info()))
            continue
        if (sample_rate is not None and sample_data is not None):
            files_read += 1
            output.append(Sample_data(labels=d["labels"], augmentations=d["augmentations"], rate=sample_rate, data=sample_data))
    if verbose:
        print("\tRead: {}/{} ({} files found).".format(files_read, files_attempted, files_found))
    return output


def check_in_file_data(file_data, hit_labels, kit_labels, tech_labels, pre_augs, post_augs):
    hit_check = set(hit_labels) == set(file_data["hit_label"])
    kit_check = set(kit_labels) == set(file_data["kit_label"])
    tech_check = set(tech_labels) == set(file_data["tech_label"])
    pre_augs_check = set(pre_augs) == set(file_data["pre_augs"])
    post_augs_check = set(post_augs) == set(file_data["post_augs"])
    if hit_check and kit_check and tech_check and pre_augs_check and post_augs_check:
        return True
    return False

'''
Saves a specific type of audio data into the correct directories based on label and augmentation.
Assumes that all data in data_set has the same sample rate
'''
def save_data(data_type, data_set, verbose=False):
    # Check all information is correct
    if verbose:
        print("Attempting to save " + data_type + " data and labels...")
    if (SETTINGS is None):
        print("\tRead failed, since settings not found.")
        return False
    if (data_type not in SETTINGS.data.keys()):
        print("\t{} is not a valid data_type.")
        return False
    # Get root path for data and the file extension from parameters.
    data_info = SETTINGS.data[data_type]
    root_path = os.path.join(os.getcwd(), data_info["path"])
    file_ext = "."+data_info["file_type"]
    
    # Categorise data_set by labels and augmentations (if any)
    file_data = []
    total_files = len(data_set)
    for data in data_set:
        hit_labels = data.labels["hit_label"]
        kit_labels = data.labels["kit_label"]
        tech_labels = data.labels["tech_label"]
        pre_augs = [aug for aug in data.augmentations if aug in PRE_AUGS]
        post_augs = [aug for aug in data.augmentations if aug in POST_AUGS]
        data_added = False
        for fd in file_data:
            if check_in_file_data(fd, hit_labels, kit_labels, tech_labels, pre_augs, post_augs):
                fd["data_set"].append(data.data)
                data_added = True
        if (data_added == False):
            file_data.append({"data_set": [data.data], "hit_label": hit_labels, "kit_label": kit_labels, "tech_label": tech_labels, "pre_augs": pre_augs, "post_augs": post_augs, "sample_rate": data.rate})

    # Save
    success_count = 0
    for fd in file_data:
        # Subdirectory logic (find correct existing from categories if one doesn't exist make it) and determine the first uid for the file(s) to be created
        subdirs = ["-".join(fd[sub]) for sub in data_info["dir_structure"]]
        path = os.path.join(root_path, *subdirs)
        uid = 0
        if (not os.path.exists(path)):
            try:
                os.makedirs(path)
            except OSError:
                if verbose:
                    print("\t\tCouldn't make new folder: ", path)
                return False
        else:
            currentfiles = list(map(lambda filename: int(filename[:-len(file_ext)]), filter(lambda filename: filename[-len(file_ext):] == file_ext, os.listdir(path))))
            if len(currentfiles) > 0:
                currentfiles.sort(reverse=True)
                uid = currentfiles[0] + 1

        # Save file logic
        for d in fd["data_set"]:
            filename = str(uid) + file_ext
            filepath = os.path.join(path, filename)
            try:
                if (file_ext == ".gz"):
                    np.savetxt(fname=filepath, X=d, header=str(fd["sample_rate"]), fmt="%.1e")
                else:
                    wav.write(filepath, rate=int(fd["sample_rate"]), data=d)
                success_count += 1
                uid += 1
            except:
                print("\t\tUnexpected error:\n\t\t", sys.exc_info())
                continue
    if verbose:
        print("\tSaved: {}/{} files.".format(success_count, total_files))
    return success_count == total_files

'''
Searches the path in the settings.json indicated by the "data_type" parameter for all files and returns their filepaths and puts the label and augmentation metadata with them.
'''
def get_file_classes(data_type):
    data_info = SETTINGS.data[data_type]
    file_data = []
    for path, subdirs, files in os.walk(data_info["path"]):
        file_ext = "."+data_info["file_type"]
        # Filter non ".file_type" files
        files = list(filter(lambda filename: filename[-len(file_ext):] == file_ext, files))
        file_count = len(files)
        if (file_count > 0):
            dir_structure_len = len(data_info["dir_structure"])
            dirs = path.split(os.sep)[-dir_structure_len:]
            # Assign labels from directory structure to a dictionary.
            classes = {data_info["dir_structure"][l]: dirs[l].split("-") for l in range(dir_structure_len)}
            labels = {k: v for k, v in classes.items() if k in LABELS}
            augmentations = {k: v for k, v in classes.items() if k in AUG_KEYS}
            file_data.extend([{"filepath": os.path.join(path, f), "labels": labels, "augmentations": augmentations} for f in files])
    return file_data

'''
Returns the kit label class with the smallest amount of files against it with the file count 
'''
def smallest_kit_class(data_type):
    file_data = get_file_classes(data_type)
    files_per_kit_label = {kl:0 for kl in SETTINGS.label["kit_label"]}
    for fd in file_data: 
        for kl in files_per_kit_label.keys():
            if (kl in fd["labels"]["kit_label"]):
                files_per_kit_label[kl] += 1
    file_counts = list(files_per_kit_label.values())
    file_counts.sort()
    lowest_file_count = file_counts[0]
    kit_label = None
    for kl, v in files_per_kit_label:
        if v == lowest_file_count:
            kit_label = kl
    return (kit_label, lowest_file_count)

'''
Pads the end of the samples by an amount of 0's so that the total amount of samples is a power of 2
'''
def pad_samples(samples):
    unpadded_len = samples.shape[0]
    padded_len = int(np.power(2, np.floor(np.log2(unpadded_len))+1))
    return np.pad(samples, ((0, padded_len-unpadded_len)), mode="constant")

'''
With a given sample_data instance, resamples it into approximately the new sample rate; returning it as a new sample_data instance.
'''
def reduce_sample_rate(data, new_rate, verbose=False):
    if int(new_rate) == int(data.rate):
        print("No rate change, skipping.")
        return data
    start_time = time.time()
    # Pad the length of the sample data to speed up resampling function
    samples = pad_samples(data.data)
    no_of_samples = len(samples)
    t = no_of_samples/data.rate
    if verbose:
        print("Resampling {}s of data: from {} samples/s to ~{} samples/s".format(str(round(t, 3)), data.rate, new_rate))
    # Calculating the new number of samples
    batch_size = int(new_rate/data.rate*no_of_samples)
    resampled_data = signal.resample(samples, batch_size)
    new_rate = round(batch_size/t, 4)
    end_time = time.time()
    if verbose:
        print("\tDone to {} samples/s! (Execution time: {}s)".format(new_rate, round(end_time-start_time, 2)))
    return Sample_data(labels={k:v for k,v in data.labels.items()}, augmentations={k:v for k,v in data.augmentations.items()}, rate=int(new_rate), data=np.float32(resampled_data))

## Onset detection stuff
'''
Converting amplitude into local energy
'''
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
    return np.float32(energy_signal)

'''
Converting amplitude into energy based novelty (a detection function)
'''
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
    return np.float32(energy_novelty)

'''
Finding the variable threshold of the detection function (ebn) using the median and the c1 & c2 constants.
'''
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
        print("\tDone. (Execution time: {}s)".format(round(end_time-start_time, 2)))
    return np.float32(detection_thresholds)

'''
Finding the indexes of onsets (where the ebn function is initially greater than the threshold function) when the functions are trimmed by an amount of samples either end of the audio.
'''
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
        if (above_threshold and g < 0 and prev_grad is not None and prev_grad > 0):
            onsets.append(i)
        prev_grad = g
    end_time = time.time()
    if verbose:
        print("\t{} onsets found. (Execution time: {}s)".format(len(onsets), round(end_time-start_time, 2)))
    return np.int32(onsets)