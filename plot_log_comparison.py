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
def plot_compare_tensorflow_logs(logs, metric="val_rounded_all_or_nothing_acc", title="Class accuracy"):
    print("reached call")
    logs = np.array(logs)
    logs = logs.flatten().tolist()
    print("processed list", logs)
    tag_axis_labels = {'loss': 'loss', 'acc': 'binary accuracy', 'accuracy': 'binary accuracy', 'binary_accuracy': 'binary accuracy', 'categorical_accuracy': 'categorical accuracy', 'rounded_all_or_nothing_acc': "AON accuracy", 'binary_crossentropy': "binary cross-entropy" , 'kullback_leibler_divergence': 'KL divergence'} 
    
    graph_legends = []
    ys = []

    for log_i in range(len(logs)):
        path = logs[log_i]
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

        current_acc = EventAccumulator(path, tf_size_guidance)
        current_acc.Reload()

        # Check if metric is present in the log file
        if metric not in current_acc.Tags()["scalars"]:
            continue

        graph_legends.append(" ".join([formatted_modelname]))
        ys.append(np.array([sc[2] for sc in current_acc.Scalars(metric)]))
        
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(111)
    print(ys)
    print(ys[0].size)
    steps = np.amax(list(map(lambda y: y.size, ys)))
    print("steps", steps)
    for log_i in range(len(ys)):
        xs = np.arange(1, ys[log_i].size+1)
        plt.plot(xs, ys[log_i], label=graph_legends[log_i])
    ax.set_xticks(np.arange(1, steps+1).tolist(), minor=False)
    plt.xlabel('epoch')
    if metric[:4] == "val_":
        metricname = "validation "
        metric = metric[4:]
    metricname += tag_axis_labels[metric] if metric in tag_axis_labels.keys() else metric
    plt.ylabel(metricname)
    loc = "lower right"
    plt.legend(loc=loc, frameon=True)
    plt.title(title)
    plt.show()
    #plt.savefig(os.path.join(os.getcwd(), "plots", re.sub("[ <>:\"/\\|?*{}]", "_", title)+".svg"))

'''
Plots each binary Tensorflow log within the 'logs' root directory
'''
if __name__ == '__main__':
    import argparse
    logs_root = os.path.join(os.getcwd(), "logs")
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', action='append',
                    dest='logs',
                    nargs="+",
                    help="The paths of logs to compare")
    parser.add_argument('-m', action='store', dest='metric',
                    help='The metric to compare the logs')
    parser.add_argument('-t', action='store', dest='title',
                    help='The graph title')
    args = parser.parse_args()
    print(args.logs, args.metric, args.title)
    if (args.logs is None):
        print("need to provide logs")
        exit
    elif(args.metric is None and args.title is not None):
        plot_compare_tensorflow_logs(args.logs, title=args.title)
    elif(args.metric is not None and args.title is None):
        plot_compare_tensorflow_logs(args.logs, metric=args.metric)
    elif(args.metric is not None and args.title is not None):
        plot_compare_tensorflow_logs(args.logs, metric=args.metric, title=args.title)
    else:
        plot_compare_tensorflow_logs(args.logs)