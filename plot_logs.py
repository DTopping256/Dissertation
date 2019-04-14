#!/usr/bin/env python
# coding: utf-8
# Adapted from: https://gist.github.com/tomrunia/1e1d383fb21841e8f144

import numpy as np
import os
import re
# https://github.com/tensorflow/tensorboard/blob/master/tensorboard/backend/event_processing/event_accumulator.py
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import pyplot as plt
from matplotlib import axes

'''
Plots all of the desired tags in log. 
'''
def plot_tensorflow_log(path):
    path = path if path[-1] != "\\" and path[-1] != "/" else path[:-1]
    modelname, date, time = os.path.split(path)[-1].split("_")
    formatted_modelname, formatted_date, formatted_time = re.sub("-", " ", modelname), re.sub("-", "/", date), re.sub("-", ":", time)

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
    desired_tags = ['loss', 'acc', 'accuracy', 'binary_accuracy', 'categorical_accuracy', 'rounded_all_or_nothing_acc', 'binary_crossentropy', 'kullback_leibler_divergence']
    tag_axis_labels = {'loss': 'loss', 'acc': 'binary accuracy', 'accuracy': 'binary accuracy', 'binary_accuracy': 'binary accuracy', 'categorical_accuracy': 'categorical accuracy', 'rounded_all_or_nothing_acc': "AON accuracy", 'binary_crossentropy': "binary cross-entropy" , 'kullback_leibler_divergence': 'KL divergence'} 
    tags = [t for t in desired_tags if t in event_acc.Tags()["scalars"]]
    training_tags, validation_tags = [], []
    for tag in tags:
        training_tags.append(event_acc.Scalars(tag))
        validation_tags.append(event_acc.Scalars('val_{}'.format(tag)))

    for tag_i in range(len(tags)):
        fig = plt.figure(figsize=(8,6))
        ax = plt.subplot(111)
        steps = len(training_tags[tag_i])
        x = np.arange(1, steps+1)
        y = np.zeros([steps, 2])
        for i in range(steps):
            y[i, 0] = training_tags[tag_i][i][2]
            y[i, 1] = validation_tags[tag_i][i][2]
        plt.plot(x, y[:,0], label='training')
        plt.plot(x, y[:,1], label='validation')
        ax.set_xticks(x.tolist(), minor=False)
        plt.xlabel('steps')
        metricname= tag_axis_labels[tags[tag_i]]
        plt.ylabel(metricname)
        loc = "lower right" if tags[tag_i] in ["acc", "accuracy", "binary_accuracy", "categorical_accuracy", "rounded_all_or_nothing_acc"] else "upper right"
        plt.legend(loc=loc, frameon=True)
        plt.title("{} - {}".format(metricname[0].upper()+metricname[1:], " ".join([formatted_modelname, formatted_date, formatted_time])))
        plt.savefig(os.path.join(os.getcwd(), "plots", "_".join([modelname, date, time, re.sub(" ", "-", metricname)])+".svg"))

'''
Plots each binary Tensorflow log within the 'logs' root directory
'''
if __name__ == '__main__':
    import argparse
    logs_root = os.path.join(os.getcwd(), "logs")
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='append',
                    dest='log_dirs',
                    default=None)
    args = parser.parse_args()
    plotted_logs = []
    for p in os.listdir(os.path.join(os.getcwd(), "plots")):
        p_name = "_".join(p.split("_")[0:3])
        if p_name not in plotted_logs:
            plotted_logs.append(p_name)
    new_logs = list(filter(lambda l: l not in plotted_logs, os.listdir(logs_root)))
    log_dirs = args.log_dirs if args.log_dirs is not None else [os.path.join("logs", d) for d in new_logs]
    for log_dir in log_dirs:
        path = os.path.join(os.getcwd(), log_dir)
        if (os.path.isdir(path)):
            plot_tensorflow_log(path)