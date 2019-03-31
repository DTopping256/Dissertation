#!/usr/bin/env python
# coding: utf-8
# Adapted from: https://gist.github.com/tomrunia/1e1d383fb21841e8f144

import numpy as np
import os
import re
# https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/event_accumulator.py
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import pyplot as plt

'''
Plots all of the desired tags in log. 
'''
def plot_tensorflow_log(path):
    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Select desired tags in the log file if they are present
    desired_tags = ['loss', 'acc', 'hamming', 'kullback_leibler_divergence']
    tag_axis_labels = {'loss': 'loss', 'acc': 'accuracy', 'hamming': 'hamming loss', 'kullback_leibler_divergence': 'KL divergence'} 
    tags = [t for t in desired_tags if t in event_acc.Tags()["scalars"]]
    training_tags, validation_tags = [], []
    for tag in tags:
        training_tags.append(event_acc.Scalars(tag))
        validation_tags.append(event_acc.Scalars('val_{}'.format(tag)))

    fig, axs = plt.subplots(len(tags), 1, constrained_layout=True, figsize=(3.5, len(tags)*2.5))
    for tag_i in range(len(tags)):
        steps = len(training_tags[tag_i])
        x = np.arange(1, steps+1)
        y = np.zeros([steps, 2])
        for i in range(steps):
            y[i, 0] = training_tags[tag_i][i][2]
            y[i, 1] = validation_tags[tag_i][i][2]
        axs[tag_i].plot(x, y[:,0], label='training')
        axs[tag_i].plot(x, y[:,1], label='validation')
        axs[tag_i].set_xticks(x.tolist(), minor=False)
        axs[tag_i].set_xlabel('steps')
        axs[tag_i].set_ylabel(tag_axis_labels[tags[tag_i]])
        loc = "lower right" if tags[tag_i] == "acc" else "upper right"
        axs[tag_i].legend(loc=loc, frameon=True)
    
    filename = os.path.split(path)[-1]
    if (re.match('(([0-9]|[A-Z]|[a-z])+-){2}([0-9]|[A-Z]|[a-z])+_.{0,}', filename)):
        model, loss_f, optim = filename.split('_')[0].split('-')
        fig.suptitle('{} {} {}'.format(model[0].upper()+model[1:], loss_f.upper(), optim.upper()))
    else:
        fig.suptitle(filename)
    plt.show()

'''
Plots each binary Tensorflow log within the 'logs' root directory
'''
if __name__ == '__main__':
    import argparse
    logs_root = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='append',
                    dest='log_dirs',
                    default=None,
                    help='Log directories, relative to {}'.format(logs_root))
    args = parser.parse_args()
    log_dirs = args.log_dirs if args.log_dirs is not None else os.listdir(logs_root)
    for log_dir in log_dirs:
        path = os.path.join(logs_root, log_dir)
        if (os.path.isdir(path)):
            plot_tensorflow_log(path)

