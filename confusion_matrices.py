#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from keras.models import model_from_json
from keras import backend as K
import tensorflow as tf
from tqdm import tqdm
import sys
import os
import re
import gc
from generator import AudioGenerator, multilabelled_ys_to_labels, onehot_superclass_labels_to_ys, MULTI_LABEL, ONE_HOT, TIME_SEQUENCE, LOG_SPECTROGRAM, LINEAR_SPECTROGRAM

sys.path.append("./modules")
from multiclass_labelling_utils import kit_combinations
from audio_utils import *

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

config = tf.ConfigProto(allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 1},
                        log_device_placement = True,
                        gpu_options=gpu_options
                       )

session = tf.Session(config=config)
K.set_session(session)

kcs_strings = list(map(lambda x: re.sub("-", " ", " & ".join(sorted(list(x)))), kit_combinations()))
kcs_strings

# Adapted from: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
models_dir = "models"

def load_model(name):
    loaded_model_json = None
    with open(os.path.join(models_dir, name+".json"), 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(models_dir, name+".h5"))
    print("Loaded model from disk")
    return loaded_model

# Adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
'''# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)'''

cm_dir = "confusion_matrices"

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          filename=None,
                          cmap=plt.cm.RdYlGn):
    if title is None:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    if filename is None:
        filename = 'confusion_matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    '''print(cm)'''

    fig, ax = plt.subplots(figsize=(30,30))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center", fontsize="x-small",
                    color="black")
    fig.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), cm_dir, filename))

# Expect numpy arrays
def onehot_to_kcs(y):
    pred = np.where(y >= 0.5)[0]
    if pred.size == 0:
        return False
    else:
        return pred[0]

def multihot_to_kcs(y):
    labels = multilabelled_ys_to_labels(y)
    if (len(labels["hit_label"])==0):
        return False
    try:
        y = onehot_superclass_labels_to_ys(labels)
    except ValueError:
        return False
    return onehot_to_kcs(y)

# Return the kcs index for a given numpy label array.
get_ind = {MULTI_LABEL: multihot_to_kcs, ONE_HOT: onehot_to_kcs}

# Gen settings
batch_size = 50
data_type = "test"

# Test data generator
def get_gen(problem_type, input_type):
    sample_metadata = get_file_classes(data_type)
    n = len(sample_metadata)
    filenames = [(sm["filepath"]) for sm in sample_metadata]
    labels = [sm["labels"] for sm in sample_metadata]
    gen = AudioGenerator(filenames, labels, data_type, batch_size, shuffle=True, problem_type = problem_type, input_type = input_type)
    # Shuffle the data
    gen.on_epoch_end()
    return (gen, n)

# Loads a model and generator, then predicts the test data set and returns y_pred and y_true as an array of superclass indexes. (disposes of model and generator in memory)
def load_and_test_model(name, problem_type, input_type):
    model = load_model(name)
    # Get test data generator
    generator, n = get_gen(problem_type, input_type)
    pred_y_inds, true_y_inds, unclassified = [], [], 0
    # Loop through however many batches until nearly all test data used
    try:
        for i in tqdm(range(n // batch_size)):
            batch_x, true_y = generator.__getitem__(i)
            pred_y = model.predict(batch_x)
            # Convert numpy arrays to kcs indexes and add them to lists
            for y_i in range(len(pred_y)):
                pred_y_ind = get_ind[problem_type](pred_y[y_i])
                if (pred_y_ind == False):
                    unclassified += 1
                    continue
                pred_y_inds.append(pred_y_ind)
                true_y_inds.append(get_ind[problem_type](true_y[y_i]))
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    finally:
        generator = None
        model = None
        gc.collect()
    return (pred_y_inds, true_y_inds, n, unclassified)

def test_data_confusion_matrices(model_name):
    model_encoding_specifiers = {"OneHot": ONE_HOT, "MultiHot": MULTI_LABEL}
    model_input_specifiers = {"(1D)": TIME_SEQUENCE, "(linear 2D)" : LINEAR_SPECTROGRAM, "(log 2D)": LOG_SPECTROGRAM}
    specifier_string, date, time = model_name.split("_") 
    specifiers = specifier_string.split("-") 
    encoding, input_type = None, None
    for specifier in specifiers:
        if specifier in model_encoding_specifiers.keys():
            encoding = model_encoding_specifiers[specifier]
        elif specifier in model_input_specifiers.keys():
            input_type = model_input_specifiers[specifier]
    y_pred, y_true, n, unclassified = load_and_test_model(model_name, encoding, input_type)
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    formatted_name = " ".join([*specifiers, re.sub("-", "/", date), re.sub("-", ":", time)])
    print("\n\n"+formatted_name)
    print("y_pred:", y_pred)
    print("\ny_true:", y_true)
    print("\nUnclassified data:", unclassified)

    with open(os.path.join(os.getcwd(), cm_dir, "log.txt"), "a") as log_file:
        log_file.write("{} | Unclassified: {} ({}%)\n".format(model_name, unclassified, round(unclassified/n, 1)))

    # Plot normalized confusion matrix
    if (y_pred.size > 0):
        plot_confusion_matrix(y_true, y_pred, classes=np.array(kcs_strings), normalize=True,
                        title='Confusion matrix - {}'.format(formatted_name), filename=model_name+".svg")
    

'''
Loads each model into memory and runs the test data on it. Then plots the results into a confusion matrix.
'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='append',
                    dest='models',
                    default=None,
                    help='Model names')
    args = parser.parse_args()
    models = args.models if args.models is not None else list(map(lambda filename: filename[:-3], filter(lambda filename: ".h5" in filename, os.listdir(models_dir))))
    existing_confusion_matrices = []
    with open(os.path.join(os.getcwd(), cm_dir, "log.txt"), "r") as log_file:
        logs = log_file.read().split("\n")
        for l in logs:
            existing_confusion_matrices.append(l.split("|")[0][:-1])

    print(existing_confusion_matrices)
    models = list(filter(lambda x: x not in existing_confusion_matrices, models))
    for model_name in models:
        test_data_confusion_matrices(model_name)