#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import collections
import numpy as np
from scipy.signal import savgol_filter, convolve, normalize
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa
import time

# Allows me to import my modules
sys.path.append('./modules')
from audio_utils import *
import data_augmentation as data_aug


# In[4]:


file_i = 0


# In[5]:


samples = read_data(data_type="cropped", file_index=file_i, batch_size=5, verbose=True)
print(len(samples) if samples is not False else 0)

augmented_samples = data_aug.pre_multiclass_augmentation(samples)

aug_stacks = data_aug.get_aug_stacks(data_aug.PRE_MULTI_AUGS)
# Create new augmentation data ordered dictionary
aug_data = collections.OrderedDict([("-".join(stack), []) for stack in aug_stacks])
aug_keys = list(aug_data.keys())

plt.plot(samples[4].data, color="blue")
plt.title("No augmentation")
plt.ylabel("Amplitude (arb. unit)")
plt.xlabel("Sample number")
plt.show()

plt.plot(samples[4].data, color="blue", label="Not augmented")
plt.plot(augmented_samples[4]["amplitude"][0], color="grey", label="Augmented")
plt.title("Amplitude augmentation")
plt.ylabel("Amplitude (arb. unit)")
plt.xlabel("Time (sample number)")
plt.legend()
plt.show()

plt.plot(samples[4].data, color="blue", label="Not augmented")
plt.plot(augmented_samples[4]["pitch"][-1], color="grey", label="Augmented")
plt.title("Pitch augmentation")
plt.ylabel("Amplitude (arb. unit)")
plt.xlabel("Time (sample number)")
plt.legend()
plt.show()

plt.plot(samples[4].data, color="blue", label="Not augmented")
plt.plot(augmented_samples[4]["translate"][3], color="grey", label="Augmented")
plt.title("Translation augmentation")
plt.ylabel("Amplitude (arb. unit)")
plt.xlabel("Time (sample number)")
plt.legend()
plt.show()